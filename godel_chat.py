import numpy as np
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


checkpoint = "microsoft/GODEL-v1_1-large-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

class ChatBot():
    def __init__(self,instruction):
        self.instruction = instruction
        self.chat_history_ids = None
        self.bot_input_ids = None
        self.end_chat = False
        self.welcome()
        
    def welcome(self):
        print("Initializing ChatBot ...")
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        time.sleep(3)
        greeting = "Welcome, I am ChatBot, here for your kind service"
        print("ChatBot >>  " + greeting)

    def user_input(self):
        # receive input from user
        text = input("User    >> ")
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            self.end_chat=True
            print('ChatBot >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting ChatBot ...')
        else:
            dialog = ' EOS '.join(text)
            query = f"{self.instruction} [CONTEXT] {dialog} "
            self.new_user_input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids

    def bot_response(self):
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1) 
        else:
            self.bot_input_ids = self.new_user_input_ids
        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
        response = tokenizer.decode(self.chat_history_ids[0], skip_special_tokens=True)
        print('ChatBot >>  '+ response)




# # build a ChatBot object
# instruction = f'Instruction: given a dialog context, you need to response empathically.'
# bot = ChatBot(instruction)
# # start chatting
# while True:
#     # receive user input
#     bot.user_input()
#     # check whether to end chat
#     if bot.end_chat:
#         break
#     # output bot response
#     bot.bot_response()    
def generate(instruction, knowledge, dialog=None):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    #dialog = input()
    #while True:
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256, min_length=8, top_p=0.9, do_sample=True)
    #print(outputs)
    dialog = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #dialog = input("Enter:")
    return dialog

# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context, you need to response empathically.'
# Leave the knowldge empty
knowledge = ''
#Hey, I am Ayush! How are you doing today?
dialog = [
    'I am also doing well'
]
response = generate(instruction, knowledge, dialog)
print(response)