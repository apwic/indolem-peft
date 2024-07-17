#!/bin/bash
for i in {0..4}
do
	echo "Training on fold $i with UniPELT"

	OUTPUT_DIR="bin/$DATASET-unipelt-$i"
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
		--weight_decay $WEIGHT_DECAY \
		--num_beams $NUM_BEAMS \
		--patience $PATIENCE \
		--optim "adamw_torch_fused" \
		--max_source_length $MAX_LENGTH \
		--max_target_length $MAX_LENGTH \
		--generation_max_length $GEN_MAX_LENGTH \
		--pad_to_max_length \
		--seed $SEED \
		--bf16 \
		--predict_with_generate \
		--evaluation_strategy "epoch" \
		--logging_strategy "epoch" \
		--save_strategy "epoch" \
		--save_total_limit 1 \
		--load_best_model_at_end \
		--metric_for_best_model "rouge1" \
		--report_to "wandb" \
		--push_to_hub \
		--project_name "indolem-pelt-$DATASET" \
		--group_name "unipelt" \
		--job_type "fold-$i" \
		--run_name "$DATASET-unipelt-$i" \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir \
		--adapter_config "unipelt" \
		--train_adapter

	echo "Finished training on fold $i with UniPELT"
done
