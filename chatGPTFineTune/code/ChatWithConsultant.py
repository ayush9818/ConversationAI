import os 
import openai 


class ChatBot(object):
  def __init__(self, api_key):
    openai.api_key = api_key
    self.history = "JOY-> Hello, I am your personal mental health assistant. What's on your mind today?\n"
    print(self.history)
    
  def start_conversation(self):
    user_input = input("User-> ")
    while user_input not in ['bye','exit']:
      self.history += f"User-> {user_input}JOY->"
      response = openai.Completion.create(
                  model="davinci:ft-personal-2023-01-14-09-05-17",
                  prompt=self.history,
                  temperature=0.89,
                  max_tokens=162,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0.6,
                  stop=["JOY","User","USER","Joy"]
                  )
      response = response['choices'][0].get('text')
      response = response.replace('END','').replace('END_TO_END','')
      answer = f"JOY-> {response}"
      self.history += f"{response}\n"
      print(answer)
      user_input = input("User-> ")


if __name__ == '__main__':
    API_KEY = "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"
    consultant = ChatBot(api_key = API_KEY)
    consultant.start_conversation()
