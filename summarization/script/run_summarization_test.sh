#!/bin/bash
MODEL="LazarusNLP/IndoNanoT5-base"
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=8
NUM_EPOCHS=1
LEARNING_RATE=5e-5
SEED=42

DATA_DIR=./data/
export DATASET=summarization

OUTPUT_DIR="bin/$DATASET-test"
TRAIN_FILE="$DATA_DIR/train.01.jsonl"
VALIDATION_FILE="$DATA_DIR/dev.01.jsonl"
TEST_FILE="$DATA_DIR/test.01.jsonl"

python run_summarization.py \
	--model_name_or_path $MODEL \
	--lang "id" \
	--text_column "paragraphs" \
	--summary_column "summary" \
	--source_prefix "summarize: " \
	--output_dir $OUTPUT_DIR \
	--train_file $TRAIN_FILE \
	--validation_file $VALIDATION_FILE \
	--test_file $TEST_FILE \
	--num_train_epochs $NUM_EPOCHS \
	--per_device_train_batch_size $TRAIN_BATCH_SIZE \
	--per_device_eval_batch_size $EVAL_BATCH_SIZE \
	--learning_rate $LEARNING_RATE \
	--seed $SEED \
	--bf16 \
	--predict_with_generate \
	--max_train_samples 50 \
	--max_predict_samples 50 \
	--max_eval_samples 50 \
	--evaluation_strategy "epoch" \
	--logging_strategy "epoch" \
	--save_strategy "epoch" \
	--save_total_limit 1 \
	--report_to "none" \
	--do_train \
	--do_eval \
	--do_predict \

