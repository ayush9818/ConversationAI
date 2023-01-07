
# ConversationAI
Semantic Search Based conversational AI 

The bot is trained on a wikipedia datasets of movie plots. Given any query about the movie plot, the bot will run the search queries to match with the closest plot and return the movie Titles.

## Installation

```bash
  git clone https://github.com/ayush9818/ConversationAI.git
  cd ConversationAI
  python3 -m venv venv 
  source venv/bin/activate
  pip install -r requirements.txt
```


    
## Run Locally

Download the model files in models directory and unzip the file

```bash
  https://drive.google.com/drive/folders/10ZYOZOmfbUpcCZXGsRYu9NGn8Nr5SvGI?usp=share_link
```

Download the dataset files in the dataset folder

```bash
  https://drive.google.com/file/d/11cHg0nKPMiCgUFjClysF7gEqa4XE7LmN/view?usp=share_link
  https://drive.google.com/file/d/1ZS14XdHT4hsj5SO-3FSmIuXFQNR9rC3P/view?usp=share_link

```

Run the search engine 
```python
  python run_search_engine.py --data-path dataset/wiki_movie_plots_deduped.csv \
   --index-path dataset/search-model_v5_100k.index \
   --pretrained-weight-file models/search-model_v5_100k \
   --topk 2
```


