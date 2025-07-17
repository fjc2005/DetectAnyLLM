for METHOD in "likelihood" "logrank" "entropy"
do
    for SUBSET in "DIG" "SIG"
    do
        for TASK in "generate" "polish" "rewrite"
        do
        accelerate launch other_method/eval_baselines.py \
            --scoring_model_name gpt-neo-2.7B \
            --cache_dir ./model/ \
            --method ${METHOD} \
            --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
            --eval_data_format MIRAGE \
            --save_path ./results/${METHOD} \
            --save_file MIRAGE_${SUBSET}_${TASK}.json \
            --eval_batch_size 8 \
            --seed 42
        done
    done
done