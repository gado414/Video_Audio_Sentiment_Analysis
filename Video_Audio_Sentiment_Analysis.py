import streamlit as st
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import soundfile as sf
import os
import librosa
import numpy as np
import cv2
from deepface import DeepFace

# Inisialisasi modul speech recognition
recognizer = sr.Recognizer()

# Inisialisasi modul Sastrawi
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Fungsi untuk membersihkan dan mengolah teks menggunakan Sastrawi
def clean_and_process_text(text):
    # Membersihkan teks dari karakter khusus
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Mengonversi teks ke huruf kecil
    cleaned_text = cleaned_text.lower()

    # Menghapus stop words menggunakan Sastrawi
    cleaned_text = stopword_remover.remove(cleaned_text)

    # Melakukan stemming menggunakan Sastrawi
    cleaned_text = stemmer.stem(cleaned_text)

    return cleaned_text

# Fungsi untuk mengubah suara menjadi teks
def voice_to_text_asli(audio_path):
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text_result = recognizer.recognize_google(audio_data, language="id-ID")
            return text_result
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# Fungsi untuk melakukan NLP pada teks
def nlp_processing(text):
    nlp = pipeline("text-classification", model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")
    result = nlp(text)
    return result

# Fungsi untuk membuat mapping antara skor sentiment dengan kategori
def map_sentiment_category(score):
    if score >= 0.6:
        return 1  # Positif
    elif score <= 0.4:
        return -1  # Negatif
    else:
        return 0  # Netral

# Fungsi untuk membuat tag cloud dan mengembalikan data kata-kata beserta frekuensinya
def create_tag_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

    # Menghitung frekuensi kata-kata
    words = text.split()
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    return word_freq

# Fungsi untuk membuat diagram kata-kata dominan
def create_word_frequency_chart(word_freq):
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    top_words = list(sorted_word_freq.keys())[:10]
    top_freqs = list(sorted_word_freq.values())[:10]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_words, top_freqs, color='skyblue')
    ax.set_xlabel('Kata')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Top 10 Kata yang Mempengaruhi Sentimen')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Fungsi untuk menampilkan hasil di bawah kategori dan membuat tag cloud serta diagram
def display_result_with_tag_cloud(text, sentiment_score, category):
    sentiment_label = ""
    if sentiment_score == 1:
        sentiment_label = "Positif"
    elif sentiment_score == -1:
        sentiment_label = "Negatif"
    else:
        sentiment_label = "Netral"
    
    result_text = f"{category}\nVoice to Text: {text}\nAnalisis Sentimen: {sentiment_score} - {sentiment_label}\n"

    # Display results in Streamlit
    st.write(result_text)

    # Create and display tag cloud
    word_freq = create_tag_cloud(text)

    # Create and display word frequency chart
    create_word_frequency_chart(word_freq)

# Fungsi untuk mengunggah file audio dengan tag cloud
def upload_audio_with_tag_cloud():
    uploaded_audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_audio_file is not None:
        # Save the uploaded file to a temporary path with the original file name
        file_extension = uploaded_audio_file.name.split('.')[-1]
        original_file_name = os.path.splitext(uploaded_audio_file.name)[0]
        audio_path = f"{original_file_name}.{file_extension}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio_file.read())

        st.success(f"Audio '{uploaded_audio_file.name}' uploaded successfully!")

        # Convert MP3 to WAV
        if file_extension == 'mp3':
            converted_audio_path = f"{original_file_name}.wav"
            audio_data, samplerate = sf.read(audio_path)
            sf.write(converted_audio_path, audio_data, samplerate)

            # Perform audio analysis
            analyze_audio(converted_audio_path, uploaded_audio_file.name)
        else:
            # Perform audio analysis
            analyze_audio(audio_path, uploaded_audio_file.name)

# Fungsi untuk melakukan analisis audio
def analyze_audio(audio_path, original_file_name):
    st.write("Analyzing audio...")

    # Perform voice to text on the audio file
    file_text = voice_to_text_asli(audio_path)

    if file_text:
        # Display the Voice to Text result
        st.write("Voice to Text result:")
        st.write(file_text)

        # Perform text processing
        st.write("Cleansing Data result:")
        cleaned_text = clean_and_process_text(file_text)
        st.write(cleaned_text)

        # Perform sentiment analysis
        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])

        # Display the result with tag cloud
        display_result_with_tag_cloud(cleaned_text, sentiment_score, f"Results from {original_file_name}")

    # Perform emotion recognition on the audio
    emotion_recognition = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    emotions = emotion_recognition(audio_path)

    st.write("Emotion Recognition Results:")
    for emotion in emotions:
        st.write(f"{emotion['label']}: {emotion['score']}")

# Fungsi untuk mengunggah file video dengan tag cloud
def upload_video_with_tag_cloud():
    uploaded_video_file = st.file_uploader("Upload Video File", type=["mp4"])

    if uploaded_video_file is not None:
        # Save the uploaded video file to a temporary location with the original file name
        original_file_name = os.path.splitext(uploaded_video_file.name)[0]
        video_path = f"{original_file_name}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video_file.read())

        st.success("Video uploaded successfully!")

        # Perform video analysis
        analyze_video(video_path)

# Fungsi untuk melakukan analisis video
def analyze_video(video_path):
    st.write("Analyzing video...")

    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Display the video duration
    st.write(f"Video Duration: {video_clip.duration} seconds")

    # Extract audio from the video
    original_file_name = os.path.splitext(video_path)[0]
    video_audio_path = f"{original_file_name}_audio.wav"
    video_clip.audio.write_audiofile(video_audio_path)

    # Perform voice to text on the extracted audio
    file_text = voice_to_text_asli(video_audio_path)

    if file_text:
        # Display the Voice to Text result
        st.write("Voice to Text result:")
        st.write(file_text)

        # Perform text processing
        st.write("Cleansing Data result:")
        cleaned_text = clean_and_process_text(file_text)
        st.write(cleaned_text)

        # Perform sentiment analysis
        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])

        # Display the result with tag cloud
        display_result_with_tag_cloud(cleaned_text, sentiment_score, f"Results from Video {original_file_name}")

    # Perform emotion recognition on the video
    st.write("Performing emotion recognition on video...")
    try:
        video_capture = cv2.VideoCapture(video_path)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            results = DeepFace.analyze(frame, actions=['emotion'])

            # Display the emotion recognition results
            for result in results:
                st.write(f"Emotion: {result['dominant_emotion']}")

        video_capture.release()
    except Exception as e:
        st.error(f"Error in video emotion recognition: {e}")

# Main Streamlit app
def main():
    st.title("Voice and Video Sentiment Analysis with Emotion Recognition")
    st.write("Upload an audio or video file for sentiment analysis and emotion recognition")

    # Upload audio file with tag cloud
    st.subheader("Audio Analysis")
    upload_audio_with_tag_cloud()

    # Upload video file with tag cloud
    st.subheader("Video Analysis")
    upload_video_with_tag_cloud()

if __name__ == "__main__":
    main()
