#!/bin/bash
declare -a reduction_factors=("16" "64")

for reduction_factor in "${reduction_factors[@]}"
do
	echo "Training with Seq Bottleneck (Adapter) on reduction_factor=${reduction_factor}"

	OUTPUT_DIR="bin/$DATASET-seq_bn-rf${reduction_factor}"

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
		--group_name "seq_bn-rf${reduction_factor}" \
		--run_name "$DATASET-seq_bn-rf${reduction_factor}" \
		--max_train_samples $MAX_TRAIN_SAMPLES \
		--max_eval_samples $MAX_EVAL_SAMPLES \
		--max_predict_samples $MAX_PREDICT_SAMPLES \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir \
		--adapter_config "seq_bn[reduction_factor=${reduction_factor}]" \
		--train_adapter

	echo "Finished training with Seq Bottleneck (Adapter) on reduction_factor=${reduction_factor}"
done
