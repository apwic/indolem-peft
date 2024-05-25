import os
import pandas as pd
import ast

def convert_tsv_to_grouped_csv(directory, file_prefix, output_filename):
    tsv_files = [f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith('.tsv')]

    data = []
    for tsv_file in tsv_files:
        file_path = os.path.join(directory, tsv_file)
        with open(file_path, 'r') as file:
            sentences = []
            labels = []
            sentence = []
            label = []
            for line in file:
                if line.strip() == "":
                    if sentence and label:
                        sentences.append(sentence)
                        labels.append(label)
                    sentence = []
                    label = []
                else:
                    word, tag = line.strip().split()
                    sentence.append(word)
                    label.append(tag)
            if sentence and label:  # append the last sentence if file doesn't end with a blank line
                sentences.append(sentence)
                labels.append(label)
            data.extend(zip(sentences, labels))

    df = pd.DataFrame(data, columns=['tokens', 'ner_tags'])
    df['id'] = df.index.astype(str)
    df = df[['id', 'tokens', 'ner_tags']]
    
    df.to_csv(directory + output_filename, index=False)

directory = './data/'

# Convert the dev, train, and test TSV files to CSV files
convert_tsv_to_grouped_csv(directory, 'dev', 'dev.csv')
convert_tsv_to_grouped_csv(directory, 'train', 'train.csv')
convert_tsv_to_grouped_csv(directory, 'test', 'test.csv')

print("Conversion completed successfully.")
