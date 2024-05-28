
#!/bin/bash
MAX_LENGTH=200
BERT_MODEL="indolem/indobert-base-uncased"
BATCH_SIZE=30
NUM_EPOCHS=1
LEARNING_RATE=5e-5
SEED=42

DATA_DIR=./data

OUTPUT_DIR="bin/sentiment-base-test"
TRAIN_FILE="$DATA_DIR/train0.csv"
VALIDATION_FILE="$DATA_DIR/dev0.csv"
TEST_FILE="$DATA_DIR/test0.csv"

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
    --seed $SEED \
    --evaluation_strategy "epoch" \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --report_to "wandb" \
    --project_name "indolem-pelt-sentiment-temp" \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
