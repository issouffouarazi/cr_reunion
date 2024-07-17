import streamlit as st
from pydub import AudioSegment

def process_audio(file):
    # Exemple de traitement audio
    audio = AudioSegment.from_file(file)
    duration = len(audio) / 1000  # Durée en secondes
    return duration

import datetime, glob, os
import subprocess
# send pipeline to GPU (when available)
import torch
import whisper
import pyannote.audio
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Audio
from pyannote.core import Segment
import main_meetsum
import ollama

import json
import wave
import contextlib
import numpy as np
import pandas as pd
import ollama
import requests

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb") #,    device=torch.device("cuda"))


# language = 'any' #@param ['any', 'English']
# model_size = 'tiny' #@param 
# model = whisper.load_model('tiny.en')


# model = whisper.load_model(model_size)
# path = "test.mp3"
# result = model.transcribe(path)
# segments = result["segments"]

def write_segments(segments, outfile):
    """write out segments to file"""
    
    def time(secs):
      return datetime.timedelta(seconds=round(secs))
    print(len(segments))
    f = open(outfile, "w")    
    f.write("[ \n ")
    for (i, segment) in enumerate(segments):
      # if i == 0 or segments[i - 1]["seek"] != segment["seek"]:
      f.write("{  \n ")
      f.write(f'  \n  "speaker" : "{str(segment["seek"])}" ,    ' )
      f.write(f'  \n  "text" : "{str(segment["text"])} " \n ' )
      if i!=len(segments)-1 : 
        f.write("}, \n ")
      else : 
        f.write("} \n ")
    f.write("\n]")
    f.close()


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


def display_conversations_from_json(json_file):
    # Charger les données JSON depuis le fichier
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialiser un dictionnaire pour stocker les conversations par locuteur
    conversations = {}
    
    # Parcourir chaque message dans les données JSON
    for message in data:
        speaker_id = message["speaker"]
        text = message["text"]
        
        # Ajouter le message au locuteur correspondant dans le dictionnaire des conversations
        if speaker_id in conversations:
            conversations[speaker_id].append(text)
        else:
            conversations[speaker_id] = [text]
    
    # Afficher les conversations par locuteur
    for speaker_id, messages in conversations.items():
        st.subheader(f"Conversation avec le locuteur {speaker_id}")
        for i, message in enumerate(messages, start=1):
            st.write(f"{i}. {message}")


def main():

    # if 'audio_uploaded' not in st.session_state:
    #     st.session_state.audio_uploaded = False

    # if 'option_selected' not in st.session_state:
    #     st.session_state.option_selected = False

    # if 'processing_done' not in st.session_state:
    #     st.session_state.processing_done = False

    # if 'transcription_done' not in st.session_state:
    #     st.session_state.transcription_done = False

    # if 'display_choice' not in st.session_state:
    #     st.session_state.display_choice = "Non"
    
    st.title("Application de traitement audio")
    
    uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["wav", "mp3", "ogg", "flac"])
    # # Appeler la fonction de traitement audio
    # duration = process_audio(uploaded_file)
    
    # st.write(f"La durée de l'audio est de {duration} secondes.")
    
    if uploaded_file is not None:
        # Afficher les informations de base du fichier uploadé
        st.audio(uploaded_file, format='audio/wav')
        path = uploaded_file.name

        model_size = st.selectbox("Choisissez une le type de model ", ['tiny', 'base', 'small', 'medium', 'large'])
        st.write(f"Option choisie : {model_size}")

        language = st.selectbox("Choisissez une le type de model ", ['any', 'English'])
        st.write(f"Option choisie : {language}")

        model_name = model_size
        if language == 'English' and model_size != 'large':
            model_name += '.en'
        

        if st.button("Lancer la retranscription de l'audio "): 
            model = whisper.load_model(model_name)

            # result = model.transcribe(path)
            # segments = result["segments"]

            json_name = 'transcript.json'
            # write_segments(segments, json_name)
            st.write(f"La retranscription de l'audio se trouve dans le fichier : transcript.json ")
            # display_choice = st.radio("Voulez-vous afficher le contenu de transcript.json ?", ("Oui", "Non"))
            # if display_choice == "Oui":
            with open(json_name, 'r') as f:
                transcript_content = json.load(f)
            st.write("Contenu de transcript.json :")
            st.json(transcript_content)
            # display_conversations_from_json(json_name)
        

        # if st.button("Afficher le texte"):
            st.write("WAAAAW")

            st.write((json_name))
            # st.write(meeting_summary(json_name))

        
        
        col1, col2 =st.columns(2)

        with col1:
            st.subheader("Conversation")
            json_name = 'transcript.json'
            with open(json_name, 'r') as f:
                transcript_content = json.load(f)
            st.write("Contenu de transcript.json :")
            st.json(transcript_content)

            

        with col2:
            if "more_stuff" not in st.session_state:
                st.session_state.more_stuff = False

            st.subheader("Resumé")



            click = st.button("Résumer la conversation")
            if click:
                st.session_state.more_stuff = True

            if st.session_state.more_stuff:
                # st.write("Doing more optional stuff")
                st.write(meeting_summary(json_name))


if __name__ == "__main__":
    main()
