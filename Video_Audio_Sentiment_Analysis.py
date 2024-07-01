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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import cv2
import face_recognition

# Inisialisasi modul speech recognition
recognizer = sr.Recognizer()

# Inisialisasi modul Sastrawi
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Load pre-trained emotion recognition model
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

# Fungsi untuk membersihkan dan mengolah teks menggunakan Sastrawi
def clean_and_process_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = stopword_remover.remove(cleaned_text)
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

# Fungsi untuk membuat tag cloud
def create_tag_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

# Fungsi untuk menampilkan hasil di bawah kategori dan membuat tag cloud
def display_result_with_tag_cloud(text, sentiment_score, category):
    sentiment_label = ""
    if sentiment_score == 1:
        sentiment_label = "Positive"
    elif sentiment_score == -1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    result_text = f"{category}\nVoice to Text: {text}\nSentiment Analysis: {sentiment_score} - {sentiment_label}\n"

    # Display results in Streamlit
    st.write(result_text)

    # Create and display tag cloud
    create_tag_cloud(text)

# Fungsi untuk mengunggah file audio dengan tag cloud
def upload_audio_with_tag_cloud():
    uploaded_audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_audio_file is not None:
        file_extension = uploaded_audio_file.name.split('.')[-1]
        original_file_name = os.path.splitext(uploaded_audio_file.name)[0]
        audio_path = f"{original_file_name}.{file_extension}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio_file.read())

        st.success(f"Audio '{uploaded_audio_file.name}' uploaded successfully!")

        if file_extension == 'mp3':
            converted_audio_path = f"{original_file_name}.wav"
            audio_data, samplerate = sf.read(audio_path)
            sf.write(converted_audio_path, audio_data, samplerate)
            analyze_audio(converted_audio_path, uploaded_audio_file.name)
        else:
            analyze_audio(audio_path, uploaded_audio_file.name)

# Fungsi untuk melakukan analisis audio
def analyze_audio(audio_path, original_file_name):
    st.write("Analyzing audio...")

    file_text = voice_to_text_asli(audio_path)

    if file_text:
        st.write("Voice to Text result:")
        st.write(file_text)

        st.write("Cleansing Data result:")
        cleaned_text = clean_and_process_text(file_text)
        st.write(cleaned_text)

        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])

        display_result_with_tag_cloud(cleaned_text, sentiment_score, f"Results from {original_file_name}")

# Fungsi untuk mendeteksi emosi dari wajah dalam video
def detect_face_emotion(frame, face_locations):
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_image = frame[top:bottom, left:right]
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_image_rgb)
        face_tensor = transforms.ToTensor()(pil_image).unsqueeze(0)
        face_emotion_output = model(face_tensor)
        emotion_score = torch.softmax(face_emotion_output.logits, dim=1)
        emotion_label = torch.argmax(emotion_score).item()

        if emotion_label == 0:
            label = "Angry"
        elif emotion_label == 1:
            label = "Disgust"
        elif emotion_label == 2:
            label = "Fear"
        elif emotion_label == 3:
            label = "Happy"
        elif emotion_label == 4:
            label = "Sad"
        elif emotion_label == 5:
            label = "Surprise"
        elif emotion_label == 6:
            label = "Neutral"
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

# Fungsi untuk melakukan analisis video
def analyze_video(video_path):
    st.write("Analyzing video...")

    video_clip = VideoFileClip(video_path)
    st.write(f"Video Duration: {video_clip.duration} seconds")

    original_file_name = os.path.splitext(video_path)[0]
    video_audio_path = f"{original_file_name}_audio.wav"
    video_clip.audio.write_audiofile(video_audio_path)

    file_text = voice_to_text_asli(video_audio_path)

    if file_text:
        st.write("Voice to Text result:")
        st.write(file_text)

        st.write("Cleansing Data result:")
        cleaned_text = clean_and_process_text(file_text)
        st.write(cleaned_text)

        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])

        display_result_with_tag_cloud(cleaned_text, sentiment_score, f"Results from Video {original_file_name}")

    st.write("Detecting face emotions in real-time...")

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face_locations = face_recognition.face_locations(frame)
        frame = detect_face_emotion(frame, face_locations)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk mengunggah file video dengan tag cloud
def upload_video_with_tag_cloud():
    uploaded_video_file = st.file_uploader("Upload Video File", type=["mp4"])

    if uploaded_video_file is not None:
        original_file_name = os.path.splitext(uploaded_video_file.name)[0]
        video_path = f"{original_file_name}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video_file.read())

        st.success("Video uploaded successfully!")
        analyze_video(video_path)

# Fungsi utama
def main():
    menu = st.sidebar.selectbox("Pilih Menu", ["Upload Audio", "Upload Video"])

    if menu == "Upload Audio":
        upload_audio_with_tag_cloud()
    elif menu == "Upload Video":
        upload_video_with_tag_cloud()

if __name__ == "__main__":
    main()
