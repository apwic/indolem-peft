#!/bin/bash
MODEL="LazarusNLP/IndoNanoT5-base"
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=32
NUM_EPOCHS=5
LEARNING_RATE=5e-5
MAX_LENGTH=128
SEED=42

DATA_DIR=./data
export DATASET=summarization

for i in {0..4}
do
	echo "Training on fold $i with Seq Bottleneck (Adapter)"

	OUTPUT_DIR="bin/$DATASET-seq_bn-$i"
	TRAIN_FILE="$DATA_DIR/train.0$((i + 1)).jsonl"
	VALIDATION_FILE="$DATA_DIR/dev.0$((i + 1)).jsonl"
	TEST_FILE="$DATA_DIR/test.0$((i + 1)).jsonl"

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
		--generation_max_length $MAX_LENGTH \
		--seed $SEED \
		--bf16 \
		--predict_with_generate \
		--evaluation_strategy "epoch" \
		--logging_strategy "epoch" \
		--save_strategy "epoch" \
		--save_total_limit 1 \
		--report_to "tensorboard" \
		--push_to_hub \
		--project_name "indolem-pelt-$DATASET" \
		--group_name "seq_bn" \
		--job_type "fold-$i" \
		--run_name "$DATASET-seq_bn-$i" \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir \
		--adapter_config "seq_bn" \
		--train_adapter

	echo "Finished training on fold $i with Seq Bottleneck (Adapter)"
done
