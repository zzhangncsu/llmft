#!/usr/bin/env bash

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port

export PROJECT_DIR=/root/llmft

# setup basic paths
export CACHE_BASE_DIR=/root/llmft/cache
export OUTPUT_DIR=/root/llmft/logfiles

# setup wandb
export WANDB_DISABLED=false
export WANDB_API_KEY=8cab6020680320fc114c896cc94035fe8ea6f51f
export WANDB_USERNAME=llama_ft_exp
export WANDB_ENTITY=llama_ft_exp
export WANDB_CACHE_DIR=$CACHE_BASE_DIR/wandb
export WANDB_CONFIG_DIR=$WANDB_CACHE_DIR

# set variables for transformers, datasets, evaluate
export TOKENIZERS_PARALLELISM=true
export HF_DATASETS_CACHE=$CACHE_BASE_DIR/hf_datasets
export HF_EVALUATE_CACHE=$CACHE_BASE_DIR/hf_evaluate
export HF_MODULES_CACHE=$CACHE_BASE_DIR/hf_modules
export HF_MODELS_CACHE=$CACHE_BASE_DIR/hf_lms
export TRANSFORMERS_CACHE=$CACHE_BASE_DIR/transformers

# set variables for torch
export TORCH_EXTENSIONS_DIR=$CACHE_BASE_DIR/torch

# create cash dirs if they don't exist yet
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_EVALUATE_CACHE
mkdir -p $HF_MODULES_CACHE
mkdir -p $HF_MODELS_CACHE
mkdir -p $TORCH_EXTENSIONS_DIR

max_train_samples=64
epochs=40
warmup_ratio=0.25
bsz=1
num_gpus=4
learning_rate=1e-5
model_name_or_path=meta-llama/Llama-2-7b-hf

# we log at the end of every epoch
logging_steps=$((max_train_samples / (bsz * num_gpus)))

# OPT target tokens
# --target_tokens "ĠYes,ĠNo" \

# Llama target tokens
# --target_tokens "▁Yes,▁No" \

# GPT-NeoX target tokens
# --target_tokens "ĠYes,ĠNo" \
# "1" "2" "3" "4" "5" "6" "7" "8" "9"

for seed in "0"
do
    for data_seed in "0"
    do
        deepspeed \
            --include localhost:0,1,2,3 \
            --master_port 60000 \
            $PROJECT_DIR/ft.py \
            --wandb_project_name mnli-deepspeed_64_sample_40_epoch \
            --wandb_group_name pattern-verbalizer-ft \
            --model_name_or_path $model_name_or_path \
            --cache_dir $HF_MODELS_CACHE \
            --task_name mnli \
            --pattern "{text1} {text2} ?" \
            --target_tokens "▁Yes,▁No" \
            --dataset_cache_dir $HF_DATASETS_CACHE \
            --overwrite_cache True \
            --max_seq_length 256 \
            --output_dir /root/zzhang/logfiles \
            --overwrite_output_dir \
            --do_train \
            --max_train_samples $max_train_samples \
            --per_device_train_batch_size $bsz \
            --gradient_accumulation_steps 1 \
            --num_train_epochs $epochs \
            --warmup_ratio $warmup_ratio \
            --logging_first_step true \
            --logging_steps $logging_steps \
            --learning_rate $learning_rate \
            --weight_decay 0.0 \
            --do_eval \
            --evaluation_strategy epoch \
            --per_device_eval_batch_size 10 \
            --eval_on_hans \
            --save_strategy no \
            --fp16 \
            --seed $seed \
            --data_seed $data_seed \
            --deepspeed $PROJECT_DIR/deepspeed_configs/ds_config_zero3.json \
            --deepspeed_stage 3 \
            --report_to "none"
    done
done