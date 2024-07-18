#!/bin/bash
export MODEL="LazarusNLP/IndoNanoT5-base"
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
export DATASET=indosum
export DATA_DIR=./data/$DATASET

source ./script/run_summarization_base.sh
source ./script/run_summarization_lora.sh
source ./script/run_summarization_pt.sh
source ./script/run_summarization_seq_bn.sh
source ./script/run_summarization_unipelt.sh
