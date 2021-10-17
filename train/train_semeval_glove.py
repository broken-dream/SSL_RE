import argparse
import os
import threading
print(os.environ["CUDA_VISIBLE_DEVICES"])
import sys
import math
import json
sys.path.append("/home/wh/SSL_RE")

import torch.cuda
import torch.optim as optim
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import logging
import numpy as np

from encoder.lstm import BiLSTMEncoder
from model.att_lstm import AttLSTM
from utils.tokenizer import GloveSentenceTokenizer
from utils.dataset import SSLDataset, SentenceClassificationDataset, collate_fn_semeval, collate_fn_ssl_semeval, collate_fn_mt_semeval
from utils.data_utils import get_word_vec_semeval
from utils.evaluation import eval
from utils.mean_teacher_utils import softmax_mse_loss, softmax_kl_loss, get_current_consistency_weight, update_ema_variables

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

data_log = open("../data/data_log.json", "w+", encoding="utf-8")
rel2id_file = open("../data/processed_semeval/semeval_rel2id.json", encoding="utf-8")
rel2id = json.load(rel2id_file)
rel2id_file.close()

id2rel= dict([(v,k) for (k,v) in rel2id.items()])
def print_data(ids, mask, label, id2word):
    ids = ids.numpy().tolist()
    mask = mask.numpy().tolist()
    label = label.numpy().tolist()
    tokens = []
    for idx in ids:
        tokens.append(id2word[idx])
    
    print("token ids: {}".format(ids), file=data_log)
    print("att mask: {}".format(mask), file=data_log)
    print("tokens: {}".format(tokens), file=data_log)
    print("label: {}({})".format(id2rel[label], label), file=data_log)
    print("---------------------------------------------------------", file=data_log)

def print_res(golden, pred):
    out_file = open("../data/pred_res.json", "w+") 
    data = dict()
    data["golden"] = golden
    data["pred"] = pred
    print(json.dumps(data), file=out_file)
    out_file.close()

def mean_teacher(args):
        # load pretrain
    word2id, word_vec = get_word_vec_semeval(args.glove_vec_path, args.word_emb_dim)
    
    # for debug
    id2word = dict([(v,k) for k,v in word2id.items()])

    # data
    tokenizer = GloveSentenceTokenizer(word2id=word2id,
                                       max_len=args.max_len,
                                       cascade=args.cascade)

    labeled_dataset = SentenceClassificationDataset(file_path=args.labeled_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)

    unlabeled_dataset = SentenceClassificationDataset(file_path=args.unlabeled_path,
                                                 rel2id_path=args.rel2id_path,
                                                 tokenizer=tokenizer)


    val_dataset = SentenceClassificationDataset(file_path=args.val_path,
                                                rel2id_path=args.rel2id_path,
                                                tokenizer=tokenizer)

    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                   sampler=RandomSampler(labeled_dataset),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   collate_fn=collate_fn_mt_semeval)

    unlabeled_dataloader = DataLoader(dataset=unlabeled_dataset,
                                     sampler=RandomSampler(unlabeled_dataset),
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     drop_last=True,
                                     collate_fn=collate_fn_mt_semeval)

    val_loader = DataLoader(dataset=val_dataset,
                            sampler=RandomSampler(val_dataset),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn_semeval)

    # model
    encoder = BiLSTMEncoder(word_vec, args)
    model = AttLSTM(encoder, args)

    ema_encoder = BiLSTMEncoder(word_vec, args)
    ema_model = AttLSTM(ema_encoder, args)

    if args.use_gpu and torch.cuda.is_available:
        model = model.to(torch.device("cuda:0"))
        ema_model = ema_model.to(torch.device("cuda:0"))

    for param in ema_model.parameters():
        param.detach_()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    
    # criterion
    class_criterion = nn.CrossEntropyLoss(size_average=False).to(torch.device("cuda:0"))
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss

    best_metric = 0
    best_epoch = 0
    args.epoch = math.ceil(args.total_steps / args.eval_step)

    train_labeled_loader = iter(labeled_dataloader)
    train_unlabeled_loader = iter(unlabeled_dataloader)

    logger.info("-------------start traing-------------")
    logger.info("train_path: {}".format(args.labeled_path))
    logger.info("number of labeled data: {}".format(args.num_labeled))
    logger.info("batch size: {}".format(args.batch_size))
    logger.info("total epochs: {}".format(args.epoch))
    logger.info("total steps: {}".format(args.total_steps))
    logger.info("--------------------------------------")
    # train
    model.zero_grad()
    model.train()
    for epoch in range(args.epoch):
        for batch_idx in tqdm(range(args.eval_step)):
            try:
                labeled_x, labeled_mask, ema_labeled_x, ema_labeled_mask, labeled_y = next(train_labeled_loader)
            except :
                train_labeled_loader = iter(labeled_dataloader)
                labeled_x, labeled_mask, ema_labeled_x, ema_labeled_mask, labeled_y = next(train_labeled_loader)

            try:
                unlabeled_x, unlabeled_mask, ema_unlabeled_x, ema_unlabeled_mask, _ = next(train_unlabeled_loader)
            except:
                train_unlabeled_loader = iter(unlabeled_dataloader)
                unlabeled_x, unlabeled_mask, ema_unlabeled_x, ema_unlabeled_mask, _ = next(train_unlabeled_loader)
            
            inputs_x = torch.cat((labeled_x, unlabeled_x), dim=0)
            inputs_mask = torch.cat((labeled_mask, unlabeled_mask), dim=0)
            ema_inputs_x = torch.cat((ema_labeled_x, ema_unlabeled_x), dim=0)
            ema_inputs_mask = torch.cat((ema_labeled_mask, ema_unlabeled_mask), dim=0)

            if args.use_gpu and torch.cuda.is_available:
                inputs_x = inputs_x.to(torch.device("cuda:0"))
                inputs_mask = inputs_mask.to(torch.device("cuda:0"))
                ema_inputs_x = ema_inputs_x.to(torch.device("cuda:0"))
                ema_inputs_mask = ema_inputs_mask.to(torch.device("cuda:0"))
                labels = labeled_y.to(torch.device("cuda:0"))


            for i in range(min(args.batch_size, 10)):
                print_data(labeled_x[i], labeled_mask[i], labeled_y[i], id2word)
                print_data(unlabeled_x[i], unlabeled_mask[i], labeled_y[i], id2word)
                # print_data(unlabeled_x[i+args.batch_size], unlabeled_mask[i+args.batch_size], labeled_y[i], id2word)
                print("*****************************************************", file=data_log)
            
            ema_logits = ema_model(ema_inputs_x, ema_inputs_mask)
            # ema_logits = ema_logits.detach().to(torch.device("cuda:0"))
            ema_logits_labeled = ema_logits[:args.batch_size]
            logits = model(inputs_x, inputs_mask)
            logits_labeled = logits[:args.batch_size]

            class_loss = class_criterion(logits_labeled, labels) / args.batch_size
            ema_class_loss = class_criterion(ema_logits_labeled, labels) / args.batch_size
            consistency_weight = get_current_consistency_weight(epoch, args)
            consistency_loss = consistency_weight * consistency_criterion(logits, ema_logits) / args.batch_size
            loss = class_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, epoch*args.eval_step + batch_idx)
        
        eval_res = test(args, val_loader, model)
        ema_eval_res = test(args, val_loader, ema_model)
        if eval_res[args.metric] > ema_eval_res[args.metric]:
            best_eval_res = eval_res
            best_eval_res["model"] = "init"
        else:
            best_eval_res = ema_eval_res
            best_eval_res["model"] = "ema"
    
        logger.info("epoch: {}".format(epoch))
        logger.info("loss: {}".format(loss))
        for k, v in best_eval_res.items():
            logger.info("{}: {}".format(k, v))

        if eval_res[args.metric] > best_metric:
            best_metric = best_eval_res[args.metric]
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save({"state_dict": model.state_dict()}, args.save_dir + "/best_checkpoint")
            logger.info("Best checkpoint")
            best_epoch = epoch
            # best_threshold = cur_threshold
        # logger.info("best_threshold: {}".format(best_threshold))

    logger.info("-------------finished-------------")
    logger.info("best {}: {}".format(args.metric, best_metric))
    logger.info("best res epoch: {}".format(best_epoch))
    data_log.close()


def pseudo_label(args):
    # load pretrain
    word2id, word_vec = get_word_vec_semeval(args.glove_vec_path, args.word_emb_dim)
    
    # for debug
    id2word = dict([(v,k) for k,v in word2id.items()])

    # data
    tokenizer = GloveSentenceTokenizer(word2id=word2id,
                                       max_len=args.max_len,
                                       cascade=args.cascade)

    labeled_dataset = SentenceClassificationDataset(file_path=args.labeled_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)

    unlabeled_dataset = SentenceClassificationDataset(file_path=args.unlabeled_path,
                                                 rel2id_path=args.rel2id_path,
                                                 tokenizer=tokenizer)


    val_dataset = SentenceClassificationDataset(file_path=args.val_path,
                                                rel2id_path=args.rel2id_path,
                                                tokenizer=tokenizer)

    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                   sampler=RandomSampler(labeled_dataset),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   collate_fn=collate_fn_semeval)

    unlabeled_dataloader = DataLoader(dataset=unlabeled_dataset,
                                     sampler=RandomSampler(unlabeled_dataset),
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     drop_last=True,
                                     collate_fn=collate_fn_semeval)

    val_loader = DataLoader(dataset=val_dataset,
                            sampler=RandomSampler(val_dataset),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn_semeval)

    # model
    encoder = BiLSTMEncoder(word_vec, args)
    model = AttLSTM(encoder, args)
    
    # use gpu
    if args.use_gpu and torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = optim.Adadelta(params=model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    shceduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    best_metric = 0
    best_epoch = 0
    args.epoch = math.ceil(args.total_steps / args.eval_step)

    train_labeled_loader = iter(labeled_dataloader)
    train_unlabeled_loader = iter(unlabeled_dataloader)

    logger.info("-------------start traing-------------")
    logger.info("train_path: {}".format(args.labeled_path))
    logger.info("number of labeled data: {}".format(args.num_labeled))
    logger.info("batch size: {}".format(args.batch_size))
    logger.info("total epochs: {}".format(args.epoch))
    logger.info("total steps: {}".format(args.total_steps))
    logger.info("--------------------------------------")
    # train
    model.zero_grad()
    model.train()
    for epoch in range(args.epoch):
        for batch_idx in tqdm(range(args.eval_step)):
            try:
                labeled_x, labeled_mask, labeled_y = next(train_labeled_loader)
            except :
                train_labeled_loader = iter(labeled_dataloader)
                labeled_x, labeled_mask, labeled_y = next(train_labeled_loader)

            try:
                unlabeled_x, unlabeled_mask, _ = next(train_unlabeled_loader)
            except:
                train_unlabeled_loader = iter(unlabeled_dataloader)
                unlabeled_x, unlabeled_mask, _ = next(train_unlabeled_loader)
            
            inputs_x = torch.cat((labeled_x, unlabeled_x), dim=0)
            inputs_mask = torch.cat((labeled_mask, unlabeled_mask), dim=0)

            for i in range(min(args.batch_size, 10)):
                print_data(labeled_x[i], labeled_mask[i], labeled_y[i], id2word)
                print_data(unlabeled_x[i], unlabeled_mask[i], labeled_y[i], id2word)
                # print_data(unlabeled_x[i+args.batch_size], unlabeled_mask[i+args.batch_size], labeled_y[i], id2word)
                print("*****************************************************", file=data_log)
            

            if args.use_gpu and torch.cuda.is_available():
                inputs_x = inputs_x.cuda()
                inputs_mask = inputs_mask.cuda()
                labeled_y = labeled_y.cuda()
            
            # print(inputs_x.dtype)
            # print(inputs_mask.dtype)

            optimizer.zero_grad()
            logits = model(inputs_x, inputs_mask)
            logits_labeled = logits[:args.batch_size]
            logits_unlabeled = logits[args.batch_size:]
            del logits

            # labeled loss
            loss_labeled = F.cross_entropy(logits_labeled, labeled_y, reduction="mean")

            # unlabeled loss
            pseudo_label = torch.softmax(logits_unlabeled.detach()/args.temperature, dim=-1)
            max_probs, unlabeled_y = torch.max(pseudo_label, dim=-1)
            data_mask = max_probs.ge(args.threshold).float()
            loss_unlabeled = (F.cross_entropy(logits_unlabeled, unlabeled_y, reduction="none")*data_mask).mean()

            if epoch >= args.warmup_epoch:
                loss = loss_labeled + args.lambda_u * loss_unlabeled
            else:
                loss = loss_labeled

            loss.backward()
            optimizer.step()
        # shceduler.step()

        eval_res = test(args, val_loader, model)
        # eval_res, cur_threshold = dev(args, val_loader, model)

        logger.info("epoch: {}".format(epoch))
        logger.info("loss: {}".format(loss))
        for k, v in eval_res.items():
            logger.info("{}: {}".format(k, v))

        if eval_res[args.metric] > best_metric:
            best_metric = eval_res[args.metric]
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save({"state_dict": model.state_dict()}, args.save_dir + "/best_checkpoint")
            logger.info("Best checkpoint")
            best_epoch = epoch
            # best_threshold = cur_threshold
        # logger.info("best_threshold: {}".format(best_threshold))

    logger.info("-------------finished-------------")
    logger.info("best {}: {}".format(args.metric, best_metric))
    logger.info("best res epoch: {}".format(best_epoch))
    data_log.close()

def train(args):
    # load pretrain
    word2id, word_vec = get_word_vec_semeval(args.glove_vec_path, args.word_emb_dim)
    
    # for debug
    id2word = dict([(v,k) for k,v in word2id.items()])

    # data
    tokenizer = GloveSentenceTokenizer(word2id=word2id,
                                       max_len=args.max_len,
                                       cascade=args.cascade)

    labeled_dataset = SentenceClassificationDataset(file_path=args.labeled_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)

    weak_dataset = SentenceClassificationDataset(file_path=args.weak_path,
                                                 rel2id_path=args.rel2id_path,
                                                 tokenizer=tokenizer)

    strong_dataset = SentenceClassificationDataset(file_path=args.strong_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)

    unlabeled_dataset = SSLDataset(strong_data=strong_dataset,
                                   weak_data=weak_dataset)

    val_dataset = SentenceClassificationDataset(file_path=args.val_path,
                                                rel2id_path=args.rel2id_path,
                                                tokenizer=tokenizer)

    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                   sampler=RandomSampler(labeled_dataset),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   collate_fn=collate_fn_semeval)

    unlabeled_dataloader = DataLoader(dataset=unlabeled_dataset,
                                     sampler=RandomSampler(unlabeled_dataset),
                                     batch_size=args.unlabeled_batch_size,
                                     num_workers=args.num_workers,
                                     drop_last=True,
                                     collate_fn=collate_fn_ssl_semeval)

    val_loader = DataLoader(dataset=val_dataset,
                            sampler=RandomSampler(val_dataset),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn_semeval)

    # model
    encoder = BiLSTMEncoder(word_vec, args)
    model = AttLSTM(encoder, args)
    
    # use gpu
    if args.use_gpu and torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = optim.Adadelta(params=model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    shceduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    best_metric = 0
    best_epoch = 0
    args.epoch = math.ceil(args.total_steps / args.eval_step)

    train_labeled_loader = iter(labeled_dataloader)
    train_unlabeled_loader = iter(unlabeled_dataloader)

    logger.info("-------------start traing-------------")
    logger.info("train_path: {}".format(args.labeled_path))
    logger.info("number of labeled data: {}".format(args.num_labeled))
    logger.info("batch size: {}".format(args.batch_size))
    logger.info("total epochs: {}".format(args.epoch))
    logger.info("total steps: {}".format(args.total_steps))
    logger.info("--------------------------------------")
    # train
    model.zero_grad()
    model.train()
    for epoch in range(args.epoch):
        for batch_idx in tqdm(range(args.eval_step)):
            try:
                labeled_x, labeled_mask, labeled_y = next(train_labeled_loader)
            except :
                train_labeled_loader = iter(labeled_dataloader)
                labeled_x, labeled_mask, labeled_y = next(train_labeled_loader)

            try:
                unlabeled_x, unlabeled_mask = next(train_unlabeled_loader)
            except:
                train_unlabeled_loader = iter(unlabeled_dataloader)
                unlabeled_x, unlabeled_mask = next(train_unlabeled_loader)
            
            inputs_x = torch.cat((labeled_x, unlabeled_x), dim=0)
            inputs_mask = torch.cat((labeled_mask, unlabeled_mask), dim=0)

            for i in range(min(args.batch_size, 10)):
                print_data(labeled_x[i], labeled_mask[i], labeled_y[i], id2word)
                print_data(unlabeled_x[i], unlabeled_mask[i], labeled_y[i], id2word)
                print_data(unlabeled_x[i+args.batch_size], unlabeled_mask[i+args.batch_size], labeled_y[i], id2word)
                print("*****************************************************", file=data_log)
            

            if args.use_gpu and torch.cuda.is_available():
                inputs_x = inputs_x.cuda()
                inputs_mask = inputs_mask.cuda()
                labeled_y = labeled_y.cuda()
            
            # print(inputs_x.dtype)
            # print(inputs_mask.dtype)

            optimizer.zero_grad()
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

            if epoch >= args.warmup_epoch:
                loss = loss_labeled + args.lambda_u * loss_unlabeled
            else:
                loss = loss_labeled

            loss.backward()
            optimizer.step()
        shceduler.step()

        # eval_res = test(args, val_loader, model)
        eval_res, cur_threshold = dev(args, val_loader, model)

        logger.info("epoch: {}".format(epoch))
        logger.info("loss: {}".format(loss))
        for k, v in eval_res.items():
            logger.info("{}: {}".format(k, v))

        if eval_res[args.metric] > best_metric:
            best_metric = eval_res[args.metric]
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save({"state_dict": model.state_dict()}, args.save_dir + "/best_checkpoint")
            logger.info("Best checkpoint")
            best_epoch = epoch
            best_threshold = cur_threshold
        logger.info("best_threshold: {}".format(best_threshold))

    logger.info("-------------finished-------------")
    logger.info("best {}: {}".format(args.metric, best_metric))
    logger.info("best res epoch: {}".format(best_epoch))
    data_log.close()

    test_dataset =  SentenceClassificationDataset(file_path=args.test_path,
                                                  rel2id_path=args.rel2id_path,
                                                  tokenizer=tokenizer)
    test_dataloader =  DataLoader(dataset=test_dataset,
                                   sampler=SequentialSampler(test_dataset),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=False,
                                   collate_fn=collate_fn_semeval)
    test_res, t = dev(args, test_dataloader, model, best_threshold=best_threshold)
    logger.info("test f1:{}".format(test_res[args.metric]))

def train_attlstm(args):
    word2id, word_vec = get_word_vec_semeval(args.glove_vec_path, args.word_emb_dim)
    
    # for debug
    id2word = dict([(v,k) for k,v in word2id.items()])

    # data
    tokenizer = GloveSentenceTokenizer(word2id=word2id,
                                       max_len=args.max_len,
                                       cascade=args.cascade)

    labeled_dataset = SentenceClassificationDataset(file_path=args.labeled_path,
                                                   rel2id_path=args.rel2id_path,
                                                   tokenizer=tokenizer)
    val_dataset = SentenceClassificationDataset(file_path=args.val_path,
                                                rel2id_path=args.rel2id_path,
                                                tokenizer=tokenizer)
    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                   sampler=RandomSampler(labeled_dataset),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   collate_fn=collate_fn_semeval)
    val_loader = DataLoader(dataset=val_dataset,
                            sampler=RandomSampler(val_dataset),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn_semeval)

    # model
    encoder = BiLSTMEncoder(word_vec, args)
    model = AttLSTM(encoder, args)
    
    # use gpu
    if args.use_gpu and torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = optim.Adadelta(params=model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_metric = 0
    best_epoch = 0
    args.epoch = 50
    logger.info("-------------start traing-------------")
    logger.info("train_path: {}".format(args.labeled_path))
    logger.info("number of labeled data: {}".format(args.num_labeled))
    logger.info("batch size: {}".format(args.batch_size))
    logger.info("total epochs: {}".format(args.epoch))
    logger.info("total steps: {}".format(args.total_steps))
    logger.info("--------------------------------------")
    for epoch in range(args.epoch):
        for labeled_x, labeled_mask, labeled_y in tqdm(labeled_dataloader):
            if args.use_gpu and torch.cuda.is_available():
                labeled_x = labeled_x.cuda()
                labeled_mask = labeled_mask.cuda()
                labeled_y = labeled_y.cuda()
            
            optimizer.zero_grad()
            logits = model(labeled_x, labeled_mask)
            loss = criterion(logits, labeled_y)
            loss.backward()
            optimizer.step()
        
        eval_res = test(args, val_loader, model)

        logger.info("epoch: {}".format(epoch))
        logger.info("loss: {}".format(loss))
        for k, v in eval_res.items():
            logger.info("{}: {}".format(k, v))

        if eval_res[args.metric] > best_metric:
            best_metric = eval_res[args.metric]
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save({"state_dict": model.state_dict()}, args.save_dir + "/best_checkpoint")
            logger.info("Best checkpoint")
            best_epoch = epoch
    
        logger.info("-------------finished-------------")
    
    logger.info("best {}: {}".format(args.metric, best_metric))
    logger.info("best res epoch: {}".format(best_epoch))
    data_log.close()

def test(args, test_loader, model):
    ground_truth = []
    pred = []
    with torch.no_grad():
        for ids, masks, labels in iter(test_loader):
            if args.use_gpu and torch.cuda.is_available():
                ids = ids.cuda()
                masks = masks.cuda()
                pred += model(ids, masks).cpu().argmax(dim=1).numpy().tolist()
            else:
                pred += model(ids, masks).argmax(dim=1).numpy().tolist()

            ground_truth += labels.numpy().tolist()

        labels = range(args.class_num)[1:]
        eval_res = eval(ground_truth, pred, labels)
        print_res(ground_truth, pred)
        return eval_res

def dev(args, dev_loader, model, best_threshold=None):
    ground_truth = []
    pred = []
    max_prob = []
    with torch.no_grad():
        for ids, masks, labels in iter(dev_loader):
            if args.use_gpu and torch.cuda.is_available():
                ids = ids.cuda()
                masks = masks.cuda()
                logits = model(ids, masks).cpu()
            else:
                logits = model(ids, masks)
            pred += logits.argmax(dim=1).numpy().tolist()
            ground_truth += labels.numpy().tolist()
            max_prob += torch.max(F.softmax(logits, dim=-1), dim=1).values.numpy().tolist()
            #max_prob += torch.max(logits, dim=1).values.numpy().tolist()
    
    #print(max_prob[:100])

    thresholds = [0.01*i for i in range(200)]
    labels = range(args.class_num)[1:]
    
    best_res = eval(ground_truth, pred, labels)

    if best_threshold is None:
        best_t = 0
        for threshold in thresholds:
            _pred = (np.asarray(max_prob, dtype=np.float32) >= threshold).astype(np.int32) * np.asarray(pred, dtype=np.int32)
            _pred = _pred.tolist()
            # print(_pred[:100])
            eval_res = eval(ground_truth, _pred, labels)
            #logger.info("cur f1: {}".format(eval_res[args.metric]))
            #logger.info("best f1: {}".format(best_res[args.metric]))
            if eval_res[args.metric] > best_res[args.metric]:
                best_res = eval_res
                best_t = threshold
                # logger.info("temp best threshold: {}".format(best_t))
        return best_res, best_t
    else:
        pred = (np.asarray(max_prob, dtype=np.float32) >= best_threshold).astype(np.int32) * np.asarray(pred, dtype=np.int32)
        pred = pred.tolist()
        eval_res = eval(ground_truth, pred, labels)
        return eval_res, best_threshold

def test_from_scratch(args):
    # load pretrain
    word2id, word_vec = get_word_vec_semeval(args.glove_vec_path, args.word_emb_dim)
    
    # for debug
    id2word = dict([(v,k) for k,v in word2id.items()])
    tokenizer = GloveSentenceTokenizer(word2id=word2id,
                                       max_len=args.max_len,)

    test_dataset = SentenceClassificationDataset(file_path=args.test_path,
                                                    rel2id_path=args.rel2id_path,
                                                    tokenizer=tokenizer)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 collate_fn=collate_fn_semeval)
    # model
    encoder = BiLSTMEncoder(word_vec, args)
    model = AttLSTM(encoder, args)

    model.load_state_dict(torch.load(args.save_dir + "/best_checkpoint")["state_dict"])
    # use gpu
    if args.use_gpu and torch.cuda.is_available():
        model.cuda()

    eval_res = test(args, test_dataloader, model)
    for k, v in eval_res.items():
        logger.info("{}: {}".format(k, v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file path
    parser.add_argument("--labeled_path", default="../data/processed_semeval/labeled.json",
                        help="labeled file path")
    parser.add_argument("--weak_path", default="../data/weak_semeval/train.json",
                        help="weak augmented file path")
    parser.add_argument("--strong_path", default="../data/strong_semeval/train.json",
                        help="weak augmented file path")
    parser.add_argument("--val_path", default="../data/processed_semeval/val.json",
                        help="val file path")
    parser.add_argument("--test_path", default="../data/processed_semeval/test.json",
                        help="test file path")
    parser.add_argument("--save_dir", default="../result/semeval",
                        help="checkpoint path")
    parser.add_argument("--glove_vec_path", default="/home/wh/pretrain/glove.6B.300d/glove.6B.300d.txt",
                        help="glove word vector path")
    parser.add_argument("--rel2id_path", default="../data/processed_semeval/semeval_rel2id.json",
                        help="relation to id map file")

    parser.add_argument("--total_steps", default=50*32, type=int,
                        help="number of total steps to run")
    parser.add_argument("--eval_step", default=32, type=int,
                        help="number of teval steps to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size")
    parser.add_argument("--unlabeled_batch_size", default=128, type=int,
                        help="batch size for unlabeled data")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers for dataloader")          
    parser.add_argument("--use_gpu", action="store_true",
                        help="whether using gpu")             
    # hyperparameter
    parser.add_argument("--lr", default=1, type=float,
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
    parser.add_argument("--hidden_size", default=300, type=int,
                        help="lstm output length")
    parser.add_argument("--num_layers", default=1, type=int,
                        help="number of lstm layers")
    parser.add_argument("--emb_dropout_value", default=0.3, type=float,
                        help="word embedding dropout value")
    parser.add_argument("--lstm_dropout_value", default=0.3, type=float,
                        help="lstm output dropout value")
    parser.add_argument("--linear_dropout_value", default=0.5, type=float,
                        help="linear dropout value")

    # dataset meta data
    parser.add_argument("--class_num", default=19, type=int,
                        help="lstm output dropout value")
    parser.add_argument("--num_labeled", default=1454, type=int,
                        help="lstm output dropout value")

    parser.add_argument("--metric", default="micro_f",
                        help="evaluation metric")


    parser.add_argument("--model", default="fixmatch",
                        help="model selection")
    parser.add_argument("--warmup_epoch", default=1, type=int,
                        help="epoch for trainging only with labeled data")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--cascade", action="store_true")

    # arguments for pseudo_label
    parser.add_argument("--unlabeled_path", default="../data/nero_processed_semeval/nero_train.json")

    # arguments for mean teacher
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--consistency", default=None, type=float)
    parser.add_argument("--consistency-type", default="mse")
    parser.add_argument("--consistency_rampup", default=5)



    args = parser.parse_args()
    if args.mode == "train":
        if args.model == "fixmatch":
            train(args)
        elif args.model == "attlstm":
            train_attlstm(args)
        elif args.model == "pseudo_label":
            pseudo_label(args)
        elif args.model == "mean_teacher":
            mean_teacher(args)
    elif args.mode == "test":
        test_from_scratch(args)


