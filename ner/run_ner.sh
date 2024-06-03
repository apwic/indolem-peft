#!/bin/bash
export DATASET=nerugm
source ./script/run_ner_base.sh
source ./script/run_ner_lora.sh
source ./script/run_ner_pt.sh
source ./script/run_ner_unipelt.sh

export DATASET=nerui
source ./script/run_ner_base.sh
source ./script/run_ner_lora.sh
source ./script/run_ner_pt.sh
source ./script/run_ner_unipelt.sh