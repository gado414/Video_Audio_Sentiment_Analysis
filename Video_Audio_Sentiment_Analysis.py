import streamlit as st
import speech_recognition as sr
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyaudio
import wave

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

# Fungsi untuk melakukan voice to text secara real-time menggunakan PyAudio
def real_time_voice():
    st.write("Silakan mulai berbicara...")
    
    # Menggunakan PyAudio untuk merekam audio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)
    
    frames = []
    timeout = 10  # Timeout set to 10 seconds

    try:
        for i in range(0, int(44100 / 1024 * timeout)):
            data = stream.read(1024)
            frames.append(data)
    except Exception as e:
        st.warning(f"Error: {e}")
    
    st.success("Voice input berhasil diterima!")
    
    # Stop the stream and close it
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    audio_path = "real_time_audio.wav"
    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Perform voice to text on the recorded audio
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
    display_result_with_tag_cloud(cleaned_text, sentiment_score, "Real-Time Voice Analysis")

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
