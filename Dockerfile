FROM python:3.10.8

WORKDIR /usr/src/app/

COPY . . 

RUN pip install -r requirements.txt 

#CMD ["python", "run_search_engine.py", "--data-path dataset/wiki_movie_plots_deduped.csv", "--index-path dataset/search-model_v5_100k.index", "--pretrained-weight-file models/search-model_v5_100k","--topk 2"]

CMD python run_search_engine.py --data-path dataset/wiki_movie_plots_deduped.csv --index-path dataset/search-model_v5_100k.index --pretrained-weight-file models/search-model_v5_100k --topk 2