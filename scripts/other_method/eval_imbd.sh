# Train and Evaluate ImBD
TRAIN_METHOD="SPO"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="gpt-neo-2.7B"

accelerate launch train.py \
    --scoring_model_name ${SCORING_MODEL} \
    --reference_model_name ${REFERENCE_MODEL} \
    --wandb True \
    --train_data_path ./data/ai_detection_500_polish.raw_data.json \
    --train_data_format ImBD \
    --eval_data_path ./data/MIRAGE_BENCH/DIG/polish.json \
    --eval_freq 5 \
    --save_freq 2 \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4

for SUBSET in "DIG" "SIG"
do
    for TASK in "generate" "polish" "rewrite"
    do
    accelerate launch eval.py \
        --pretrained_model_name_or_path ./ckpt/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_ai_detection_500_polish.raw_data_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e5 \
        --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
        --wandb True \
        --train_method ${TRAIN_METHOD} \
        --eval_batch_size 4 \
        --save_dir ./results/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_ai_detection_500_polish.raw_data \
        --save_file MIRAGE_${SUBSET}_${TASK}.json
    done
done