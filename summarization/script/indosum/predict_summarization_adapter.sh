#!/bin/bash
for i in {0..0}
do
	echo "Predicting on fold $i"

	OUTPUT_DIR="bin/$DATASET-base-$i"
	TRAIN_FILE="$DATA_DIR/train.0$((i + 1)).jsonl"
	VALIDATION_FILE="$DATA_DIR/dev.0$((i + 1)).jsonl"
	TEST_FILE="$DATA_DIR/test.0$((i + 1)).jsonl"

	python run_summarization.py \
		--model_name_or_path $MODEL \
		--lang $LANG \
		--text_column $TEXT_COLUMN \
		--summary_column $SUMMARY_COLUMN \
		--source_prefix $SOURCE_PREFIX \
		--output_dir $OUTPUT_DIR \
		--train_file $TRAIN_FILE \
		--validation_file $VALIDATION_FILE \
		--test_file $TEST_FILE \
		--max_source_length $MAX_LENGTH \
		--max_target_length $MAX_LENGTH \
		--generation_max_length $GEN_MAX_LENGTH \
		--pad_to_max_length \
		--seed $SEED \
		--bf16 \
		--predict_with_generate \
		--report_to "none" \
		--do_predict \
		--overwrite_output_dir \
		--load_adapter $ADAPTER \
		--train_adapter 

	echo "Finished predicting on fold $i"
done
