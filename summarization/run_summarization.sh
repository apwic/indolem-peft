#!/bin/bash
export MODEL="LazarusNLP/IndoNanoT5-base"
export TRAIN_BATCH_SIZE=4
export EVAL_BATCH_SIZE=16
export NUM_EPOCHS=5
export LEARNING_RATE=5e-5
export MAX_LENGTH=128
export SEED=42
export DATA_DIR=./data
export DATASET=summarization

source ./script/run_summarization_base.sh
source ./script/run_summarization_lora.sh
source ./script/run_summarization_pt.sh
source ./script/run_summarization_seq_bn.sh
source ./script/run_summarization_unipelt.sh
