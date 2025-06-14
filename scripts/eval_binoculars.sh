for SUBSET in "DIG" "SIG"
do
    for TASK in "generate" "polish" "rewrite"
    do
    python other_method/eval_binoculars.py \
    --observer_model_path ./model/Qwen2-0.5B \
    --performer_model_path ./model/Qwen2-0.5B \
    --use_bfloat16 True \
    --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
    --eval_data_format MIRAGE \
    --save_path ./results/binoculars \
    --save_file MIRAGE_${SUBSET}_${TASK}.json \
    --eval_batch_size 8
    done
done