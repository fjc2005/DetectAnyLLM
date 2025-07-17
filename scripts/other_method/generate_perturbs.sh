PERTURB_MODEL=t5-small
N_PERTURBATIONS=100

for SUBSET in "DIG" "SIG"
do
    for TASK in "generate" "polish" "rewrite"
    do
    accelerate launch other_method/generate_perturbs.py \
        --perturb_model_name ${PERTURB_MODEL} \
        --cache_dir ./model/ \
        --data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
        --data_format MIRAGE \
        --save_path ./data/perturbed_data/${PERTURB_MODEL}_${N_PERTURBATIONS}_perturbations \
        --save_file MIRAGE_${SUBSET}_${TASK}.json \
        --seed 42 \
    done
done