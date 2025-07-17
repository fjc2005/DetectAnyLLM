for SUBSET in "DIG" "SIG"
do
    for TASK in "generate" "polish" "rewrite"
    do
    accelerate launch other_method/eval_binoculars.py \
        --observer_model_path ./model/falcon-7b \
        --performer_model_path ./model/falcon-7b-instruct \
        --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
        --eval_data_format MIRAGE \
        --save_path ./results/binoculars \
        --save_file MIRAGE_${SUBSET}_${TASK}.json \
        --eval_batch_size 8 \
        --seed 42
    done
done