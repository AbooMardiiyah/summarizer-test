
import streamlit as st
import openai
import whisper
from preprocess import preprocessor
import os
import ffmpeg
import subprocess

model = whisper.load_model('base')
api_key = st.secrets["api_key"]
openai.api_key=api_key

def summarize_video(url):
    # Transcribe the video using Whisper
    path=preprocessor(url)
    # x=url
    # path="yt.mp4"
    response = model.transcribe(path,fp16=False)
    transcription = response["text"]
    # print(transcription)
    # Summarize the transcription using GPT-3
    summarized_text=""
    text_chunks = [transcription[i:i+2048] for i in range(0, len(transcription), 2048)]
    for text_chunk in text_chunks:
        response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f'summarize this text: {text_chunk}',
        max_tokens=50,
        temperature=0
        )
        summarized_text += response["choices"][0]["text"]

    return summarized_text

st.title("YouTube Video Summarizer")

url = st.text_input("Enter the YouTube video URL:")

if url:
    summarized_text = summarize_video(url)
    st.write("Summarized text:", summarized_text)

