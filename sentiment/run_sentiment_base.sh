#!/bin/bash
export WANDB_PROJECT="indolem-pelt"
MAX_LENGTH=200
BERT_MODEL="indolem/indobert-base-uncased"
BATCH_SIZE=30
NUM_EPOCHS=20
LEARNING_RATE=5e-5
SEED=1

DATA_DIR=./data

for i in {0..0}
do
    echo "Training on fold $i"

    OUTPUT_DIR="bin/sentiment-base-$i"
    TRAIN_FILE="$DATA_DIR/train$i.csv"
    VALIDATION_FILE="$DATA_DIR/dev$i.csv"
    TEST_FILE="$DATA_DIR/test$i.csv"

    # Run the model training and evaluation
    python run_sentiment.py \
        --model_name_or_path $BERT_MODEL \
        --label_names "labels" \
        --output_dir $OUTPUT_DIR \
        --train_file $TRAIN_FILE \
        --validation_file $VALIDATION_FILE \
        --test_file $TEST_FILE \
        --max_seq_length  $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --evaluation_strategy "epoch" \
        --logging_strategy "epoch" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --report_to "tensorboard" "wandb" \
        --push_to_hub \
        --run_name "sentiment-base" \
        --do_train \
        --do_eval \
        --do_predict \
        --overwrite_output_dir

    echo "Finished training on fold $i"
done
