#!/bin/bash
LOAD_RAW_DATASET=False
if [ ${LOAD_RAW_DATASET} = "True" ]; then
  LOAD_RAW_DATASET="--load_raw_dataset"
  DATA_PATHS="[/path/dataset1,/path/dataset2]"
  DATA_WEIGHTS="[1.,1.]"
  DATA_SPLIT="90,10,0"
  SHUFFLE_BEFORE_SPLIT=""
  USE_RANDOM_SAMPLER=""
  USE_WEIGHTED_LOSS=""
  WEIGHT_BY_NUM_DOCUMENTS=""
else
  LOAD_RAW_DATASET=""
  DATA_PATHS="[/path/dataset1,/path/dataset2]"
  DATA_WEIGHTS="[1.,1.]"
  DATA_SPLIT="90,10,0"
  SHUFFLE_BEFORE_SPLIT="--shuffle_before_split"
  USE_RANDOM_SAMPLER="--use_random_sampler"
  USE_WEIGHTED_LOSS="--use_weighted_loss"
  WEIGHT_BY_NUM_DOCUMENTS="--weight_by_num_documents"
fi

#USE_XFORMERS=True
#if [ ${USE_XFORMERS} = "True" ]; then
#  USE_XFORMERS="--use_xformers"
#else
#  USE_XFORMERS=""
#fi

VOCAB_FILE="../tools/codegpt-13b-tokenizer.json"
TOKENIZER_TYPE="HFTokenizer"
MODEL_TYPE="gpt_neox"
MODEL_CONFIG_PATH="../model/${MODEL_TYPE}"

RESUME_FROM_CHECKPOINT="false"

PER_DEVICE_BATCH_SIZE=$1
TP=$2
DP=$3
EPOCH=$4
TOTAL_TRAIN_BATCH_SIZE=$(($PER_DEVICE_BATCH_SIZE * $TP * $DP))

GPU=$(($TP * $DP))
OUTPUT="/output/checkpoint/mpt-fsdp-tp-bm-${TOTAL_TRAIN_BATCH_SIZE}-tp-${TP}-dp-${DP}-gpu-${GPU}-bin"
TENSORBOARD_PATH="/output/tensorboard/mpt-fsdp-tp-bm-${TOTAL_TRAIN_BATCH_SIZE}-tp-${TP}-dp-${DP}-gpu-${GPU}-bin"

PREFIX="master-0"
mkdir -p $OUTPUT || true
echo "output to $OUTPUT"
mkdir -p $TENSORBOARD_PATH
chmod 777 $OUTPUT
chmod 777 $TENSORBOARD_PATH

# atorch environment maybe not available yet, opensource soon
pip install -U --no-deps atorch==0.1.7rc9
pip install tensorboard==2.3.0
pip install peft==0.3.0 --no-dependencies
pip install zstandard
pip install ujson
pip install jsonlines
pip list

python -m atorch.distributed.launch \
    --nproc_per_node=$(nvidia-smi -L | wc -l) \
    run_train.py \
    ${LOAD_RAW_DATASET} \
    ${SPLIT_BEFORE_READ} \
    --tokenize_mode 'pretrain' \
    --train_mode 'sst' \
    --config_path $MODEL_CONFIG_PATH \
    --tokenizer_type $TOKENIZER_TYPE \
    --vocab_file $VOCAB_FILE \
    --model_type $MODEL_TYPE \
    --padding \
    --data_paths $DATA_PATHS \
    --data_weights $DATA_WEIGHTS \
    --data_split $DATA_SPLIT \
    ${SHUFFLE_BEFORE_SPLIT} \
    ${USE_RANDOM_SAMPLER} \
    ${USE_WEIGHTED_LOSS} \
    ${WEIGHT_BY_NUM_DOCUMENTS} \
    --train_iters 100 \
    --num_warmup_steps 6000 \
    --custom_lr_scheduler_type 'cosine' \
    --learning_rate 1.0e-4 \
    --min_lr 1.0e-5 \
    --valid_iters 500 \
    --valid_interval 2000 \
    --num_train_epochs $EPOCH \
    --seq_length 512 \
    --total_train_batch_size $TOTAL_TRAIN_BATCH_SIZE \
    --per_device_valid_batch_size $PER_DEVICE_BATCH_SIZE \
    --seed 42 \
    --preprocessing_num_workers 6 \
    --num_workers 8 \
    --output_dir $OUTPUT \
    --tensorboard_dir $TENSORBOARD_PATH \
    --ignore_mismatched_sizes \
    --skip_atorch_autoacc_dryrun \
    --tp $TP \
    --dp $DP \
    --bf16 \
    --pipe_parallel_size 0 \
    --model_parallel_size 1 \
    --checkpointing_steps 5000 \
    --log_interval 10 \
    --make_vocab_size_divisible_by 128 \
    --weighted_loss_mode 'case3' \
    --checkpoint_activations \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    --max_grad_norm 1 \
    --evaluation_strategy "steps,epoch" \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --extra_save_by_epoch \
    --metric_for_best_model 'loss' \
    --greater_is_better 'false' \
    --zero_opt_level zero3 2>&1 | tee $OUTPUT/$PREFIX-output.txt