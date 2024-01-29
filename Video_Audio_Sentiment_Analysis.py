import streamlit as st
import speech_recognition as sr
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

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
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text_result = recognizer.recognize_google(audio_data, language="id-ID")
        return text_result

# Fungsi untuk melakukan NLP pada teks
def nlp_processing(text):
    nlp = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
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
        audio_path = uploaded_audio_file.name
        file_text = voice_to_text_asli(audio_path)
        st.write("Hasil Voice to Text:")
        st.write(file_text)
        st.write("\nHasil Cleansing Data:")
        cleaned_text = clean_and_process_text(file_text)
        st.write(cleaned_text)
        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])
        display_result_with_tag_cloud(cleaned_text, sentiment_score, "Results from Audio")

# Fungsi untuk mengunggah file video dengan tag cloud
def upload_video_with_tag_cloud():
    uploaded_video_file = st.file_uploader("Upload Video File", type=["mp4"])

    if uploaded_video_file is not None:
        video_path = uploaded_video_file.name
        video_clip = VideoFileClip(video_path)
        video_audio_path = "video_audio.wav"
        video_clip.audio.write_audiofile(video_audio_path)
        file_text = voice_to_text_asli(video_audio_path)
        st.write("Hasil Voice to Text:")
        st.write(file_text)
        st.write("\nHasil Cleansing Data:")
        cleaned_text = clean_and_process_text(file_text)
        st.write(cleaned_text)
        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])
        display_result_with_tag_cloud(cleaned_text, sentiment_score, "Results from Video")

# Fungsi untuk melakukan voice to text secara real-time
def real_time_voice():
    with sr.Microphone() as source:
        st.write("Silakan mulai berbicara...")
        audio_data = recognizer.listen(source, timeout=10)
        real_time_text = recognizer.recognize_google(audio_data, language="id-ID")
        st.write("Hasil Voice to Text (Asli):")
        st.write(real_time_text)
        st.write("\nHasil Cleansing Data:")
        cleaned_text = clean_and_process_text(real_time_text)
        st.write(cleaned_text)
        nlp_result = nlp_processing(cleaned_text)
        sentiment_score = map_sentiment_category(nlp_result[0]['score'])
        display_result_with_tag_cloud(cleaned_text, sentiment_score, "Results from Real-Time Voice")

# Fungsi utama
def main():
    # Pilihan menu
    menu = st.sidebar.selectbox("Pilih Menu", ["Upload Audio", "Upload Video", "Real-Time Voice"])

    # Jalankan fungsi sesuai pilihan menu
    if menu == "Upload Audio":
        upload_audio_with_tag_cloud()
    elif menu == "Upload Video":
        upload_video_with_tag_cloud()
    elif menu == "Real-Time Voice":
        real_time_voice()

if __name__ == "__main__":
    main()
