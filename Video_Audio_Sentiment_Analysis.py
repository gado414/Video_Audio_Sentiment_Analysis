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
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio_file.read())

        st.success("Audio uploaded successfully!")

        # Perform audio analysis
        analyze_audio(audio_path)

# Fungsi untuk melakukan analisis audio
def analyze_audio(audio_path):
    st.write("Analyzing audio...")

    # Perform voice to text on the uploaded audio
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
    uploaded_video_file = st.file_uploader("Upload Video File", type=["mp4"])

    if uploaded_video_file is not None:
        # Save the uploaded video file to a temporary location
        with st.spinner("Uploading video..."):
            video_path = "temp_video.mp4"
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
    video_audio_path = "video_audio.wav"
    video_clip.audio.write_audiofile(video_audio_path)

    # Perform voice to text on the extracted audio
    file_text = voice_to_text_asli(video_audio_path)

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
    display_result_with_tag_cloud(cleaned_text, sentiment_score, "Results from Video")

# Fungsi untuk melakukan voice to text secara real-time
def real_time_voice():
    webrtc_ctx = webrtc_streamer(key="sample", audio=True, video=False)

    if not webrtc_ctx.state.playing:
        st.warning("Waiting for webcam to start...")

    with st.spinner("Waiting for voice input..."):
        if webrtc_ctx.audio_recorder:
            st.write("Silakan mulai berbicara...")

            try:
                audio_data = webrtc_ctx.audio_recorder.record(timeout=10)
                st.success("Voice input berhasil diterima!")

                # Use the common function for analyzing audio
                analyze_audio_stream(audio_data)

            except sr.UnknownValueError:
                st.warning("Tidak dapat mendeteksi suara. Silakan coba lagi.")

            except sr.RequestError as e:
                st.error(f"Terjadi kesalahan pada layanan pengenalan suara: {e}")

# Fungsi untuk melakukan analisis audio dari rekaman suara real-time
def analyze_audio_stream(audio_data):
    st.write("Analyzing audio...")

    # Perform voice to text on the audio data
    file_text = recognizer.recognize_google(audio_data, language="id-ID")

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
