#!/bin/bash
export WANDB_PROJECT="indolem-pelt"
MAX_LENGTH=200
BERT_MODEL="indolem/indobert-base-uncased"
BATCH_SIZE=30
NUM_EPOCHS=20
LEARNING_RATE=5e-5
SEED=1

DATA_DIR=./data

# Define hyperparameter arrays
declare -a ranks=("2" "4" "8")
declare -a alphas=("0" "1" "2")
declare -a dropouts=("0.05" "0.1" "0.15")

for i in {0..0}
do
    for rank in "${ranks[@]}"
    do
        for alpha in "${alphas[@]}"
        do
            for dropout in "${dropouts[@]}"
            do
                echo "Training on fold $i with LoRA r=$rank, alpha=$alpha, dropout=$dropout"
                
                OUTPUT_DIR="bin/sentiment-lora-r${rank}a${alpha}d${dropout}-$i"
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
                    --max_seq_length $MAX_LENGTH \
                    --num_train_epochs $NUM_EPOCHS \
                    --per_device_train_batch_size $BATCH_SIZE \
                    --learning_rate $LEARNING_RATE \
                    --evaluation_strategy "epoch" \
                    --logging_strategy "epoch" \
                    --save_strategy "epoch" \
                    --save_total_limit 1 \
                    --report_to "tensorboard" "wandb" \
                    --push_to_hub \
                    --run_name "sentiment-lora-r${rank}a${alpha}d${dropout}-$i" \
                    --do_train \
                    --do_eval \
                    --do_predict \
                    --overwrite_output_dir \
                    --adapter_config "lora[r=$rank,alpha=$alpha,dropout=$dropout]" \
                    --adapter_name "sentiment-lora" \
                    --train_adapter \

                echo "Finished training on fold $i with LoRA r=$rank, alpha=$alpha, dropout=$dropout"
            done
        done
    done
done
