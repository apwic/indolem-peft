#!/bin/bash
export MODEL="LazarusNLP/IndoNanoT5-base"
export LANG=id
export TRAIN_BATCH_SIZE=4
export EVAL_BATCH_SIZE=8
export NUM_EPOCHS=5
export LEARNING_RATE=1e-3
export MAX_LENGTH=512
export GEN_MAX_LENGTH=128
export NUM_BEAMS=5
export WEIGHT_DECAY=0.01
export PATIENCE=5
export SEED=42
export DATASET=liputan6
export DATA_DIR=./data/$DATASET
export DATASET_NAME=id_liputan6
export DATASET_CONFIG_NAME=canonical
export TEXT_COLUMN="clean_article"
export SUMMARY_COLUMN="clean_summary"
export SOURCE_PREFIX="summarize: "

source ./script/$DATASET/run_summarization_base.sh
source ./script/$DATASET/run_summarization_lora.sh
source ./script/$DATASET/run_summarization_pt.sh
source ./script/$DATASET/run_summarization_seq_bn.sh
source ./script/$DATASET/run_summarization_unipelt.sh
