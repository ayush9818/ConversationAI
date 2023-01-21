import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time
import numpy as np 
from tqdm import tqdm
import sys
import argparse
import os
from sentence_transformers import models
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch import nn
import torch 

## Use this script to finetune the distilbert model on custom datasets

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
word_emb = models.Transformer('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')
pooling = models.Pooling(word_emb.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_emb, pooling])
model=model.to(device)

BATCH_SIZE=16
num_epochs = 1

def get_dataloader(data_path):
    """
    Read and process the input data and create a pytorch dataloader to train the model
    params:
        data_path : tsv file containing the generated queries along with their corresponding documents
    output:
        train_dataloader : pytorch dataloader for finetuning the network
    """
    train_examples = [] 
    with open(data_path) as fIn:
        for line in fIn:
            try:
                query, paragraph = line.strip().split('\t', maxsplit=1)
                train_examples.append(InputExample(texts=[query, paragraph]))
            except:
                pass
    # For the MultipleNegativesRankingLoss, it is important
    # that the batch does not contain duplicate entries, i.e.
    # no two equal queries and no two equal paragraphs.
    # To ensure this, we use a special data loader
    train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=BATCH_SIZE)
    return train_dataloader

def main(data_path, save_path):
    """
    params:
        data_path: tsv file containing the generated queries along with their corresponding documents
        save_path : path to save the trained model
    """
    train_dataloader = get_dataloader(data_path)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)

    model.save(save_path)   


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-path", help='dataset path')
  parser.add_argument("--model-save-path", default=None, help='path to save the model')
  args = parser.parse_args()


  data_path = args.data_path
  save_path = args.save_path

  os.makedirs(os.path.dirname(save_path),exist_ok=True)
  main(data_path, save_path)



