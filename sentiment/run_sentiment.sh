#!/bin/bash
export MAX_LENGTH=200
export MODEL="indolem/indobert-base-uncased"
export BATCH_SIZE=30
export NUM_EPOCHS=20
export LEARNING_RATE=5e-5
export SEED=42
export DATA_DIR=./data
export DATASET=sentiment
export LABEL_NAMES="labels"

source ./script/run_sentiment_base.sh
source ./script/run_sentiment_lora.sh
source ./script/run_sentiment_pt.sh
source ./script/run_sentiment_unipelt.sh
source ./script/run_sentiment_seq_bn.sh
