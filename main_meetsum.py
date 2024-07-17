import json
import ollama
import requests

def load_conversation_data(json_name):
  with open(json_name) as f:
      json_file = json.load(f)
      extraction = lambda x: f"{x['speaker']}: {x['text']}"
      conversation = list(map(extraction, json_file))
      conversation_string = "\n".join(conversation)
      return conversation_string
  
def meeting_summary(json_name):
     conversation_string = load_conversation_data(json_name)
     response = ollama.chat(model='llama3', messages=[
          {
          'role': 'system',
          'content': """
          Ton objectif est de résumer le texte qui vous est proposé en environ 300 mots. 
          Il s'agit d'une réunion entre une ou plusieurs personnes. Ne produisez que le résumé sans texte supplémentaire. 
          Concentrez-vous sur la rédaction d'un résumé en texte libre, avec un résumé de ce que les personnes ont dit et des actions à entreprendre.
          """
          },
     {
          'role': 'user',
          'content': conversation_string,
     },
     ])
     return response ['message']['content']
  
def meeting_summary_rest(json_name):
     conversation_string = load_conversation_data(json_name)

     prompt = """Your goal is to summarize the text that is given to you 
     in roughly 300 words. It is from a meeting between one or more people. 
     Only output the summary without any additional text. 
     Focus on providing a summary in freeform text with a summary of what 
     people said and the action items coming out of it."""

     OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"

     OLLAMA_PROMPT = f"{prompt}: {conversation_string}"
     OLLAMA_DATA = {
          "model": "llama3",
          "prompt": OLLAMA_PROMPT,
          "stream": False,
          "keep_alive": "1m",
     }

     response = requests.post(OLLAMA_ENDPOINT, json=OLLAMA_DATA)
     return response.json()["response"]

# print("Summary using the library:")
# print(meeting_summary())
# print("---------------------------")
# print("Summary using the REST API:")
# print(meeting_summary_rest())