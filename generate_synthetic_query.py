import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import time
import numpy as np 
from tqdm import tqdm
import sys
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch 

batch_size = 16 #Batch size
num_queries = 5 #Number of queries to generate for every paragraph
max_length_paragraph = 512 #Max length for paragraph
max_length_query = 64   #Max length for output query
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model = model.to(device)

def _removeNonAscii(s): return "".join(i for i in s if ord(i) < 128)

def generate_synthetic_queries(df, query_save_path):
  paragraphs = df['Plot'].tolist()
  with open(query_save_path, 'w') as fOut:
    for start_idx in tqdm(range(0, len(paragraphs), batch_size)):
        sub_paragraphs = paragraphs[start_idx:start_idx+batch_size]
        inputs = tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph, truncation=True, return_tensors='pt').to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_length_query,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=num_queries)

        for idx, out in enumerate(outputs):
            query = tokenizer.decode(out, skip_special_tokens=True)
            query = _removeNonAscii(query)
            para = sub_paragraphs[int(idx/num_queries)]
            para = _removeNonAscii(para)
            fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-path", help='dataset path')
  parser.add_argument("--query-save-path", help='path to save generated queries')
  args = parser.parse_args()

  data_path = args.data_path
  query_save_path = args.query_save_path

  df = pd.read_csv(data_path)
  generate_synthetic_queries(df, query_save_path)