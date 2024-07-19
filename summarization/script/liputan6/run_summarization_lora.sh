#!/bin/bash
declare -a ranks=("8" "16")

for rank in "${ranks[@]}"
do
	echo "Training with LoRA r=$rank"

	OUTPUT_DIR="bin/$DATASET-lora"

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
		--evaluation_strategy "epoch" \
		--logging_strategy "epoch" \
		--save_strategy "epoch" \
		--save_total_limit 1 \
		--load_best_model_at_end \
		--metric_for_best_model "rouge1" \
		--report_to "wandb" \
		--push_to_hub \
		--project_name "indolem-pelt-$DATASET" \
		--group_name "lora-r${rank}" \
		--run_name "$DATASET-lora-r${rank}" \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir \
		--adapter_config "lora[r=$rank]" \
		--train_adapter

	echo "Finished training with LoRA r=$rank"
done
