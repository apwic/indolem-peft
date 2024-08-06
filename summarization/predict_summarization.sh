#!/bin/bash
export LANG=id
export MAX_LENGTH=512
export GEN_MAX_LENGTH=128
export SEED=42
export DATASET=indosum
export DATA_DIR=./data/$DATASET
export TEXT_COLUMN="paragraphs"
export SUMMARY_COLUMN="summary"
export SOURCE_PREFIX="summarize: "

export MODEL="apwic/indosum-base-0"
source ./script/$DATASET/predict_summarization_base.sh

export MODEL="LazarusNLP/IndoNanoT5-base"
export ADAPTER="apwic/indosum-unipelt-0"
source ./script/$DATASET/predict_summarization_adapter.sh
