PERTURB_MODEL=t5-small
N_PERTURBATIONS=100

for SUBSET in "DIG" "SIG"
do
    for TASK in "generate" "polish" "rewrite"
    do
    accelerate launch other_method/eval_detect_gpt.py \
        --scoring_model_name gpt-neo-2.7B \
        --cache_dir ./model/ \
        --eval_data_path ./data/perturbed_data/${PERTURB_MODEL}_${N_PERTURBATIONS}_perturbations/MIRAGE_${SUBSET}_${TASK}.json \
        --save_path ./results/DetectGPT \
        --save_file MIRAGE_${SUBSET}_${TASK}.json \
        --eval_batch_size 8 \
        --seed 42
    done
done