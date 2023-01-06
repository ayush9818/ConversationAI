import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time
import numpy as np 
from tqdm import tqdm
import sys
import argparse
import os

def create_index(model, df, out_path): 
  encoded_data = model.encode(df.Plot.tolist())
  encoded_data = np.asarray(encoded_data.astype('float32'))
  index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
  index.add_with_ids(encoded_data, np.array(range(0, len(df))))
  faiss.write_index(index, out_path)
  return index


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-path", help='dataset path')
  parser.add_argument("--index-save-path", help='index file save path')
  parser.add_argument("--pretrained-weight-file", default=None, help='pretrained weights path')

  args = parser.parse_args()
  data_path = args.data_path
  index_save_path = args.index_save_path
  pretrained_weight_file = args.pretrained_weight_file


  df = pd.read_csv(data_path)

  if pretrained_weight_file is None:
    model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
  else:
    model = SentenceTransformer(pretrained_weight_file)

  print(f"Creating Index for Dataset")
  create_index(model, df , index_save_path)
