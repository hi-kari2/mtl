#!_dev_scores_epoch.json/bin/bash
if [[ $# -ne 3 ]]; then
  echo "train.sh <batch_size> <grad_acc_steps> <gpu>"
  exit 1
fi
prefix="all"
BATCH_SIZE=16
GRAD_ACC_STEPS=2
gpu=0
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="mcduration,mcorder,mctype,mcfrequency,mcstationarity"
test_datasets="mcduration,mcorder,mctype,mcfrequency,mcstationarity"

MODEL_ROOT="checkpoints"
BERT_PATH="xlm-roberta-large"
DATA_DIR="dataset/dvd_ppf_new/0/xlm-roberta-large/"

answer_opt=0
optim="adam"
grad_clipping=0
global_grad_clipping=1
lr="1e-5"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"

python train.py  --epochs 10  --task_def experiments/mcall_task_def.yml --encoder_type 5  --data_dir ${DATA_DIR} --multi_gpu_on  --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr}
