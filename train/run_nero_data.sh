CUDA_VISIBLE_DEVICES=$1 python -W ignore train_semeval_glove.py\
    --use_gpu\
    --batch_size=50\
    --unlabeled_batch_size=100\
    --labeled_path=../data/processed_semeval/processed_hard_match.json\
    --weak_path=../data/nero_processed_semeval/weak_clean.json\
    --strong_path=../data/nero_processed_semeval/strong_clean.json\
    --val_path=../data/nero_processed_semeval/nero_dev.json\
    --test_path=../data/nero_processed_semeval/nero_test.json\
    --save_dir=../result/semeval_nero/$1\
    --model=fixmatch
