
# GPT3 based ChatConsultant

A GPT3 based chatconsultant that can address queries on depression.

## Installation

Repository Setup

```bash
    git clone https://github.com/ayush9818/ConversationAI.git
    cd ConversationAI/openai_chatgpt3_chatconsultant
```
    
## Run using Python

Virtual Environment Setup

```bash
  python3 -m venv venv 
  source venv/bin/activate
  pip install -r requirements.txt
```

## Dataset Preparation
```bash
    python prepare_dataset.py --data-path dataset/Mental_Health_FAQ.csv \
    --save-path dataset/depression.jsonl
```

## Training using OPENAI 

Create an API Key from [API_KEY](https://beta.openai.com/account/api-keys)

Prepare dataset using OpenAI
```bash
    openai tools fine_tunes.prepare_data -f dataset/depression.jsonl
```

Set API KEY as an Environment Variable
```bash
    set OPENAI_API_KEY=<API_KEY>
```

Start Training of the Model
```bash
    openai api fine_tunes.create -t dataset/depression_prepared.jsonl -m davinci
```

The command prompt will generate a model name, save the model name somewhere.

## Chat with Consultant
```bash
    python ChatWithConsultant.py --api-key <API_KEY> \
    --model-name <MODEL_NAME>
```


## Authors

- [Ayush Agarwal](https://www.github.com/ayush9818)

