#!/bin/bash
export MODEL="indolem/indobert-base-uncased"
export TRAIN_BATCH_SIZE=16
export EVAL_BATCH_SIZE=64
export NUM_EPOCHS=100
export LEARNING_RATE=5e-5
export MAX_LENGTH=128
export SEED=42

export DATASET=nerugm
export DATA_DIR=./data/$DATASET
source ./script/run_ner_base.sh
source ./script/run_ner_lora.sh
source ./script/run_ner_pt.sh
source ./script/run_ner_unipelt.sh
source ./script/run_ner_seq_bn.sh

export DATASET=nerui
export DATA_DIR=./data/$DATASET
source ./script/run_ner_base.sh
source ./script/run_ner_lora.sh
source ./script/run_ner_pt.sh
source ./script/run_ner_unipelt.sh
source ./script/run_ner_seq_bn.sh
