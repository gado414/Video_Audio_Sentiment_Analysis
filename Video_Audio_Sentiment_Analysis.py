import streamlit as st
import speech_recognition as sr
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from streamlit_webrtc import webrtc_streamer  # Added import for webrtc_streamer

# Inisialisasi modul speech recognition
recognizer = sr.Recognizer()

# Inisialisasi modul Sastrawi
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Fungsi untuk membersihkan dan mengolah teks menggunakan Sastrawi
def clean_and_process_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = stopword_remover.remove(cleaned_text)
    cleaned_text = stemmer.stem(cleaned_text)
    return cleaned_text

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

# Fungsi untuk melakukan real-time voice-to-text dan analisis sentimen
def real_time_voice():
    # Use webrtc_streamer to stream webcam video
    webrtc_ctx = webrtc_streamer(key="sample")

    if not webrtc_ctx.state.playing:
        st.warning("Waiting for webcam to start...")

    with st.spinner("Waiting for voice input..."):
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            st.write("Silakan mulai berbicara...")

            try:
                audio_data = recognizer.listen(source, timeout=10)
                st.success("Voice input berhasil diterima!")

                st.write("Hasil Voice to Text:")
                real_time_text = recognizer.recognize_google(audio_data, language="id-ID")
                st.write(real_time_text)

                st.write("Hasil Cleansing Data:")
                cleaned_text = clean_and_process_text(real_time_text)
                st.write(cleaned_text)

                st.write("Analyzing sentiment...")
                nlp_result = nlp_processing(cleaned_text)
                sentiment_score = map_sentiment_category(nlp_result[0]['score'])

                st.write("Sentiment Analysis Result:")
                st.write(f"Score: {nlp_result[0]['score']}")
                st.write(f"Sentiment Label: {nlp_result[0]['label']}")

                # Display the result with tag cloud
                display_result_with_tag_cloud(cleaned_text, sentiment_score, "Real-Time Voice Analysis")

            except sr.UnknownValueError:
                st.warning("Tidak dapat mendeteksi suara. Silakan coba lagi.")

            except sr.RequestError as e:
                st.error(f"Terjadi kesalahan pada layanan pengenalan suara: {e}")

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
