export DATA_DIR=./data/nerugm

for i in {0..4}
do
    cat $DATA_DIR/train$i.tsv | tr '\t' ' '  | tr '  ' ' ' > $DATA_DIR/train.txt
    cat $DATA_DIR/dev$i.tsv  | tr '\t' ' '  | tr '  ' ' ' > $DATA_DIR/dev.txt
    cat $DATA_DIR/test$i.tsv  | tr '\t' ' '  | tr '  ' ' ' > $DATA_DIR/test.txt
done

cat $DATA_DIR/train.txt $DATA_DIR/dev.txt $DATA_DIR/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt