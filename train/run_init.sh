CUDA_VISIBLE_DEVICES=3 python -W ignore train_semeval_glove.py\
    --use_gpu\
    --labeled_path=../data/processed_semeval/processed_hard_match.json\
    --batch_size=50\
    # --mode=test\