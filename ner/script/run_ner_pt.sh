#!/bin/bash
BERT_MODEL="indolem/indobert-base-uncased"
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=20
LEARNING_RATE=5e-5
MAX_LENGTH=128
SEED=42
DATASET=nerui
DATA_DIR=./data/$DATASET

TRAIN_FILE="$DATA_DIR/train.csv"
VALIDATION_FILE="$DATA_DIR/dev.csv"
TEST_FILE="$DATA_DIR/test.csv"

declare -a prefix_lengths=("10" "20" "30")

for prefix_length in "${prefix_lengths[@]}"
do
    echo "Training with Prefix-Tuning prefix_length=$prefix_length"
    
    OUTPUT_DIR="bin/$DATASET-pt-pl${prefix_length}"

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
        --project_name "indolem-pelt-$DATASET-temp" \
        --run_name "$DATASET-pt-pl${prefix_length}" \
        --do_train \
        --do_eval \
        --do_predict \
        --overwrite_output_dir \
        --adapter_config "prefix_tuning[prefix_length=$prefix_length]" \
        --adapter_name "$DATASET-pt" \
        --train_adapter \

    echo "Finished training with Prefix-Tuning prefix_length=$prefix_length"
done
