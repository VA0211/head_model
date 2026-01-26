
#!/usr/bin/env bash

fold=$1
input_dir=$2
working_dir=$3
result_dir=$4

# Define the specific sub-folder for this fold
input_dir_fold=${input_dir}/${fold}

if [ -d ${working_dir} ]; then
    rm -rf ${working_dir}
fi

if [ ! -d ${result_dir} ]; then
    mkdir -p ${result_dir}
fi

# --- THE CRITICAL CHANGE ---
# We removed the '&' at the end of the python command.
# This forces the computer to finish Loop 0 before starting Loop 1.
# We also keep --gpu=0 to use RTX 4050.

for i in 0 1 2 3 4
do
    echo "Starting training for seed ${i}..."
    python full.py \
	    --gpu=-1 \
	    --input_dir=$input_dir_fold \
	    --working_dir=$working_dir/$i \
	    --seeds=$i \
	    > ${result_dir}/train_log_${i}.txt 2>&1
    echo "Finished seed ${i}."
done

# Evaluate the results
echo "Evaluating..."
python make_submission.py \
	--mean=geometric \
	--out ${result_dir}/submission.txt \
	--out-probability ${result_dir}/probability.npy \
	${working_dir}/*/*/lev2_xgboost2/test.h5 \
	> ${result_dir}/make_submission_log.txt 2>&1

python evaluate.py \
	--prediction ${result_dir}/submission.txt \
	--ground-truth ${input_dir_fold}/test/labels.txt \
	-l ${input_dir_fold}/label_names.txt \
	-o ${result_dir}/ \
	> ${result_dir}/evaluate_log.txt 2>&1

echo "Done! Check ${result_dir}/evaluate_log.txt for accuracy."