import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time
import numpy as np 
from tqdm import tqdm
import sys
import argparse

def fetch_movie_info(df, dataframe_idx):
  info = df.iloc[dataframe_idx]
  meta_dict = dict()
  meta_dict['Title'] = info['Title']
  meta_dict['Plot'] = info['Plot'][:500]
  return meta_dict

def search(df, query, top_k, index, model):
  t=time.time()
  query_vector = model.encode([query])
  top_k = index.search(query_vector, top_k)
  print('>>>> Results in Total Time: {}'.format(time.time()-t))
  top_k_ids = top_k[1].tolist()[0]
  top_k_ids = list(np.unique(top_k_ids))
  results =  [fetch_movie_info(df, idx) for idx in top_k_ids]
  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-path", help='dataset path')
  parser.add_argument("--index-path", help='index file  path')
  parser.add_argument("--pretrained-weight-file", default=None, help='pretrained weights path')
  args = parser.parse_args()

  data_path = args.data_path
  index_path = args.index_path
  pretrained_weight_file = args.pretrained_weight_file


  df = pd.read_csv(data_path)
  index = faiss.read_index(index_path)
  if pretrained_weight_file is None:
    model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
  else:
    model = SentenceTransformer(pretrained_weight_file)

  while True:
    status = input(f"Enter <q> to exit <c> for continue\n")
    if status.lower() == "q":
      print(f"Stopping Search Engine")
      break
    
    query = input(f"Enter a Search Query to find movie recommendations\n")
    results=search(df, query, top_k=5, index=index, model=model)
    print("\n")
    for result in results:
      print('\t',f"Title : {result['Title']} , Plot : {result['Plot']}")

    print("\n")
