#!/bin/bash

OUTPUT_DIR="bin/$DATASET-base"

python run_summarization.py \
	--model_name_or_path $MODEL \
	--lang "id" \
	--text_column $TEXT_COLUMN \
	--summary_column $SUMMARY_COLUMN \
	--source_prefix $SOURCE_PREFIX \
	--output_dir $OUTPUT_DIR \
	--dataset_name  $DATASET_NAME \
	--dataset_config_name $DATASET_CONFIG_NAME \
	--dataset_data_dir $DATA_DIR\
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
	--predict_with_generate \
	--report_to "none" \
	--max_predict_samples $MAX_PREDICT_SAMPLES \
	--do_predict \
	--adapter_config "unipelt" \
	--overwrite_output_dir

