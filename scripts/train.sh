# main experiment
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"

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
