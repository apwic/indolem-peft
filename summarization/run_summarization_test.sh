#!/bin/bash
BERT_MODEL="indolem/indobert-base-uncased"
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=5
LEARNING_RATE=5e-5
MAX_LENGTH=128
SEED=42

DATA_DIR=./data/
export DATASET=summarization

for i in {0..4}
do
    echo "Training on fold $i"

    OUTPUT_DIR="bin/$DATASET-base-$i"
    TRAIN_FILE="$DATA_DIR/train.0$i.jsonl"
    VALIDATION_FILE="$DATA_DIR/dev.0$i.jsonl"
    TEST_FILE="$DATA_DIR/test.0$i.jsonl"

    python run_ner.py \
		--model_name_or_path $BERT_MODEL \
		--text_column "paragraphs" \
		--summary_column "summary" \
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
		--return_entity_level_metrics \
		--evaluation_strategy "epoch" \
		--logging_strategy "epoch" \
		--save_strategy "epoch" \
		--save_total_limit 1 \
		--report_to "none" \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir

    echo "Finished training on fold $i"
done
