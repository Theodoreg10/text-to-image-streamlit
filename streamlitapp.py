import pyttsx3
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import streamlit as st


model=pickle.load(open('finalized_model.sav', 'rb'))

engine=pyttsx3.init()
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.tessa')

#with open("img_embed.txt", 'r') as f:
   # img_embed = [line.rstrip('\n') for line in f]
with open("img_embed", "rb") as fp:
    img_embed = pickle.load(fp)

with open("img_names", "rb") as fp:
    img_names = pickle.load(fp)


def speak(text, rate=100):
    engine.setProperty('rate', rate)
    engine.say(text)
    #engine.runAndWait()

def listen():
    return

def main():
    return

def search(query, k=3):
  speak(query)
  query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=True)
  print(query_emb)

  hits=util.semantic_search(query_emb , img_embed , top_k=k)

  print("Query : ", query)
  for hit in hits[0]:
    img_path=img_names[hit['corpus_id']]
    im=Image.open(img_path)
    st.image(im)

text=st.text_input("descibe you image","flower",placeholder="flower")
search(text, k=3)