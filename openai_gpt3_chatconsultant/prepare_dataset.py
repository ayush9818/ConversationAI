import pandas as pd
import json
import os
import argparse

## Randomly select 100 samples, so the we don't exceed free training quota of OpenAI
NUM_SAMPLES = 100

def prepare_dataset(data_path, save_path):
    """
    - Prepares dataset for OpenAI ChatGPT training.
    - Each entry in the dataset contains a prompt which is a question and a compeletion which is an answer
    params:
        data_path : input csv path
        save_path : path to save the prepared dataset
    """
    df = pd.read_csv(data_path)
    df = df.sample(n=NUM_SAMPLES)
    dataset = []
    for idx,row in df.iterrows():
        _data = {}
        _data['prompt'] = row['Questions']
        _data['completion'] = row['Answers']
        dataset.append(_data)
    with open(save_path, 'w') as outfile:
        for line in dataset:
            json.dump(line, outfile)
            outfile.write('\n')
    print(f"Saved dataset to {save_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help='raw dataset path')
    parser.add_argument("save-path",help="path to save the prepared dataset")
    args = parser.parse_args()
    
    data_path = args.data_path
    save_path = args.save_path
    
    prepare_dataset(data_path, save_path)
    
    