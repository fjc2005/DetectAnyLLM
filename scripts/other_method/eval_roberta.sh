for MODEL in "roberta-base-openai-detector" "roberta-large-openai-detector"
do
    for SUBSET in "DIG" "SIG"
    do
        for TASK in "generate" "polish" "rewrite"
        do
        accelerate launch other_method/eval_roberta.py \
            --model_name ${MODEL} \
            --cache_dir ./model/ \
            --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
            --eval_data_format MIRAGE \
            --save_path ./results/${MODEL} \
            --save_file MIRAGE_${SUBSET}_${TASK}.json \
            --eval_batch_size 8 \
            --seed 42
        done
    done
done