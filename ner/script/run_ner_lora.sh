#!/bin/bash
export WANDB_PROJECT="indolem-pelt-nerugm-temp"
BERT_MODEL="indolem/indobert-base-uncased"
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=20
LEARNING_RATE=5e-5
MAX_LENGTH=128
SEED=42
DATA_DIR=./data/nerugm

TRAIN_FILE="$DATA_DIR/train.csv"
VALIDATION_FILE="$DATA_DIR/dev.csv"
TEST_FILE="$DATA_DIR/test.csv"

declare -a ranks=("2" "4" "8")
declare -a alphas=("0" "1" "2")
declare -a dropouts=("0.05" "0.1" "0.15")

for rank in "${ranks[@]}"
do
    for alpha in "${alphas[@]}"
    do
        for dropout in "${dropouts[@]}"
        do
            echo "Training with LoRA r=$rank, alpha=$alpha, dropout=$dropout"
            
            OUTPUT_DIR="bin/nerugm-lora-r${rank}a${alpha}d${dropout}"

            python run_ner.py \
                --model_name_or_path $BERT_MODEL \
                --label_names "labels" \
                --text_column_name "tokens" \
                --label_column_name "ner_tags" \
                --output_dir $OUTPUT_DIR \
                --train_file $TRAIN_FILE \
                --validation_file $VALIDATION_FILE \
                --test_file $TEST_FILE \
                --num_train_epochs $NUM_EPOCHS \
                --per_device_train_batch_size $TRAIN_BATCH_SIZE \
                --per_device_eval_batch_size $EVAL_BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --max_seq_length $MAX_LENGTH \
                --seed $SEED \
                --evaluation_strategy "epoch" \
                --logging_strategy "epoch" \
                --save_strategy "epoch" \
                --save_total_limit 1 \
                --report_to "tensorboard" "wandb" \
                --push_to_hub \
                --project_name "indolem-pelt-nerugm-temp" \
                --run_name "nerugm-lora-r${rank}a${alpha}d${dropout}" \
                --do_train \
                --do_eval \
                --do_predict \
                --overwrite_output_dir \
                --adapter_config "lora[r=$rank,alpha=$alpha,dropout=$dropout]" \
                --adapter_name "nerugm-lora" \
                --train_adapter \

            echo "Finished training with LoRA r=$rank, alpha=$alpha, dropout=$dropout"
        done
    done
done
