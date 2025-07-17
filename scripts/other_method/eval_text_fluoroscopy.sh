for SUBSET in "DIG" "SIG"
do
    for TASK in "generate" "polish" "rewrite"
    do
    accelerate launch other_method/eval_text_fluoroscopy.py \
        --pretrained_model_name_or_path ./model/gte-Qwen1.5-7B-instruct \
        --clf_model_path ./model/TextFluoroscopy \
        --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
        --eval_data_format MIRAGE \
        --save_path ./results/TextFluoroscopy \
        --save_file MIRAGE_${SUBSET}_${TASK}.json \
        --eval_batch_size 8 \
        --seed 42
    done
done