import argparse
import os

import torch.cuda
import torch.optim as optim
from torch.utils.data import RandomSampler, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging

from encoder.lstm import BiLSTMEncoder
from model.att_lstm import AttLSTM
from utils.tokenizer import GloveSentenceTokenizer
from utils.dataset import SSLDataset, SentenceClassificationDataset
from utils.data_utils import get_word_vec_semeval
from utils.evaluation import eval

logger = logging.getLogger(__name__)


def train(args):
    # data
    tokenizer = GloveSentenceTokenizer(vocab_path=args.vocab_path,
                                       max_len=args.max_len)

    labeld_dataset = SentenceClassificationDataset(file_path=args.labeld_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)

    weak_dataset = SentenceClassificationDataset(file_path=args.weak_path,
                                                 rel2id_path=args.rel2id_path,
                                                 tokenizer=tokenizer)

    strong_dataset = SentenceClassificationDataset(file_path=args.strong_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)

    unlabeld_dataset = SSLDataset(strong_data=strong_dataset,
                                  weak_data=weak_dataset)

    val_dataset = SentenceClassificationDataset(file_path=args.val_path,
                                                rel2id_path=args.rel2id_path,
                                                tokenizer=tokenizer)

    labeld_dataloader = DataLoader(dataset=labeld_dataset,
                                   sampler=RandomSampler(labeld_dataset),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True)

    unlabeld_dataloader = DataLoader(dataset=unlabeld_dataset,
                                     sampler=RandomSampler(unlabeld_dataset),
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            sampler=RandomSampler(val_dataset),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    # model
    word_vec = get_word_vec_semeval(args.glove_vec_path)
    encoder = BiLSTMEncoder(word_vec, args)
    model = AttLSTM(encoder, args)

    # use gpu
    if args.use_gpu and torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr,
                           weight_decay=1e-2)

    best_metric = 0
    best_epoch = 0

    train_labeled_loader = iter(labeld_dataloader)
    train_unlabeled_loader = iter(unlabeld_dataloader)

    logger.info("-------------start traing-------------")
    logger.info("train_path: {}".format(args.labeld_path))
    logger.info("number of labeled data: {}".format(args.num_labled))
    logger.info("batch size: {}".format(args.batch_size))
    logger.info("total epochs: {}".format(args.epoch))
    logger.info("total steps: {}".format(args.total_steps))
    logger.info("--------------------------------------")
    # train
    model.zero_grad()
    model.train()
    for epoch in range(args.epoch):
        for batch_idx in tqdm(range(args.eval_steps)):
            try:
                labeled_x, labeled_mask, labeled_y = next(train_labeled_loader)
            except :
                train_labeled_loader = iter(labeld_dataloader)
                labeled_x, labeled_mask, labeled_y = next(train_labeled_loader)

            try:
                unlabeled_x, unlabeled_mask = next(train_unlabeled_loader)
            except:
                train_unlabeled_loader = iter(unlabeld_dataloader)
                unlabeled_x, unlabeled_mask = next(train_unlabeled_loader)

            inputs_x = torch.cat(labeled_x, labeled_mask)
            inputs_mask = torch.cat(labeled_mask, unlabeled_mask)
            logits = model(inputs_x, inputs_mask)
            logits_labeled = logits[:args.batch_size]
            logits_strong, logits_weak = logits[args.batch_size:].chunk(2)
            del logits

            # labeled loss
            loss_labeled = F.cross_entropy(logits_labeled, labeled_y, reduction="mean")

            # unlabeled loss
            pseudo_label = torch.softmax(logits_weak.detach()/args.temperature, dim=-1)
            max_probs, unlabeled_y = torch.max(pseudo_label, dim=-1)
            data_mask = max_probs.ge(args.threshold).float()
            loss_unlabeled = (F.cross_entropy(logits_strong, unlabeled_y, reduction="none")*data_mask).mean()

            loss = loss_labeled + args.lambda_u * loss_unlabeled
            loss.backward()
            optimizer.step()
            model.zero_grad()

        eval_res = test(args, val_loader, model, epoch)

        logger.info("epoch: {}".format(epoch))
        for k, v in eval_res.items():
            logger.info("{}: {}".format(k, v))

        if eval_res[args.metric] > best_metric:
            best_metric = eval_res[args.metric]
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save({"state_dict": model.state_dict()}, args.save_path)
            logger.info("Best checkpoint")
            best_epoch = epoch

        logger.info("--------------------------------------")

    logger.info("-------------finished-------------")
    logger.info("best {}: {}".format(args.metric, best_metric))
    logger.info("best res epoch: {}".format(best_epoch))


def test(args, test_loader, model, epoch):
    ground_truth = []
    pred = []
    with torch.no_grad():
        for ids, masks, labels in iter(test_loader):
            if args.use_gpu and torch.cuda.is_available():
                ids = ids.cuda()
                masks = masks.cuda()
                pred += model(ids, masks).cpu().argmax(dim=1).numpy.tolist()
            else:
                pred += model(ids, masks).argmax(dim=1).numpy.tolist()

            ground_truth += labels.numpy().tolist()

        labels = range(args.class_num)[1:]
        eval_res = eval(ground_truth, pred, labels)
        return eval_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file path
    parser.add_argument("--labeled_path", default="../data/strong_semeval/labeled.json",
                        help="labeled file path")
    parser.add_argument("--weak_path", default="../data/weak_semeval/train.json",
                        help="weak augmented file path")
    parser.add_argument("--strong_path", default="../data/strong_semeval/train.json",
                        help="weak augmented file path")
    parser.add_argument("--val_path", default="../data/processed_semeval/val.json",
                        help="val file path")
    parser.add_argument("--vocab_path", default="/home/wh/pretrain/glove.6B.300d/word2id.6B.json",
                        help="glove word id map file")
    parser.add_argument("--save_path", default="../result/semeval/nest_checkpoint",
                        help="checkpoint path")
    parser.add_argument("--word_vec_path", default="/home/wh/pretrain/glove.6B.300d/glove.6B.300d.npy",
                        help="glove word vector path")
    parser.add_argument("--rel2id_path", default="../data/processed_semeval/semeval_rel2id.json",
                        help="relation to id map file")

    parser.add_argument("--total_steps", default=2**20, type=int,
                        help="number of total steps to run")
    parser.add_argument("--eval_step", default=1024, type=int,
                        help="number of teval steps to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size")
    # hyperparameter
    parser.add_argument("--lr", default=1e-1, type=float,
                        help="learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="L2 coefficient")
    parser.add_argument("--lambda_u", default=1, type=float,
                        help="unlabeled loss coefficient")
    parser.add_argument("--temperature", default=1, type=float,
                        help="temperature for sharpen")
    parser.add_argument("--threshold", default=0.95, type=float,
                        help="threshold for pesudo label")

    # model parameter
    parser.add_argument("--max_len", default=128, type=int,
                        help="maximal sequence length")
    parser.add_argument("--word_emb_dim", default=300, type=int,
                        help="word embedding dimension")
    parser.add_argument("--hidden_size", default=100, type=int,
                        help="lstm output length")
    parser.add_argument("--num_layers", default=1, type=int,
                        help="number of lstm layers")
    parser.add_argument("--emb_dropout_value", default=0.3, type=float,
                        help="word embedding dropout value")
    parser.add_argument("--emb_dropout_value", default=0.3, type=float,
                        help="word embedding dropout value")
    parser.add_argument("--lstm_dropout_value", default=0.3, type=float,
                        help="lstm output dropout value")
    parser.add_argument("--linear_dropout_value", default=0.5, type=float,
                        help="linear dropout value")
    parser.add_argument("--lstm_dropout_value", default=0.3, type=float,
                        help="lstm output dropout value")
    parser.add_argument("--lstm_dropout_value", default=0.3, type=float,
                        help="lstm output dropout value")

    # dataset meta data
    parser.add_argument("--class_num", default=19, type=int,
                        help="lstm output dropout value")
    parser.add_argument("--num_labeled", default=1454, type=int,
                        help="lstm output dropout value")



