export DATASET=nerugm
export DATA_DIR=./data/$DATASET

cat $DATA_DIR/$DATASET.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt
