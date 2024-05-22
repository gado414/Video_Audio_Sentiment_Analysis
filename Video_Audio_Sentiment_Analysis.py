import streamlit as st
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import speech_recognition as sr

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
    nlp = pipeline("text-classification", model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")
    result = nlp(text)
    return result

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
        # Save the uploaded file to a temporary path
        file_extension = uploaded_audio_file.name.split('.')[-1]
        audio_path = "temp_audio." + file_extension
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio_file.read())

        st.success(f"Audio '{uploaded_audio_file.name}' uploaded successfully!")

        # Perform audio analysis
        analyze_audio(audio_path)

# Fungsi untuk melakukan analisis audio
def analyze_audio(audio_path):
    st.write("Analyzing audio...")

    # Perform voice to text on the audio file
    file_text = voice_to_text_asli(audio_path)

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
    display_result_with_tag_cloud(cleaned_text, sentiment_score, "Results from Audio")

# Fungsi untuk mengunggah file video dengan tag cloud
def upload_video_with_tag_cloud():
    st.write("Video analysis is not supported in this version.")

# Fungsi utama
def main():
    # Pilihan menu
    menu = st.sidebar.selectbox("Pilih Menu", ["Upload Audio"])

    # Jalankan fungsi sesuai pilihan menu
    if menu == "Upload Audio":
        upload_audio_with_tag_cloud()

if __name__ == "__main__":
    main()
