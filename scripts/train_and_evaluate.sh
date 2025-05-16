# Train all
for TRAIN_METHOD in "DDL" "SPO"
do
    for BASE_MODEL in "Qwen2-0.5B" "gpt-neo-2.7B"
    do
    accelerate launch train.py \
    --scoring_model_name ${BASE_MODEL} \
    --wandb True \
    --train_data_path ./data/ai_detection_500_polish.raw_data.json \
    --train_data_format ImBD \
    --eval_data_path ./data/MIRAGE_BENCH/DIG/polish.json \
    --eval_freq 1 \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4
    done
done


# Evaluate all
for TRAIN_METHOD in "DDL" "SPO"
do
    for BASE_MODEL in "Qwen2-0.5B" "gpt-neo-2.7B"
    do
        for SUBSET in "DIG" "SIG"
        do
            for TASK in "generate" "polish" "rewrite"
            do
            accelerate launch eval.py \
            --pretrained_model_name_or_path ./ckpt/${TRAIN_METHOD}_${BASE_MODEL}_ai_detection_500_polish.raw_data_e5_lr0.0001_bs1_beta0.05_r8/ \
            --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
            --wandb True \
            --train_method ${TRAIN_METHOD} \
            --eval_batch_size 4 \
            --save_dir ./results/MIRAGE_${SUBSET}
            done
        done
    done
done