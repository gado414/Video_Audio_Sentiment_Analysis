import os
import re
import subprocess
import tempfile

import imageio_ffmpeg
import matplotlib.pyplot as plt
import speech_recognition as sr
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from wordcloud import WordCloud

MODEL_NAME = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
CHUNK_SECONDS = 50  
@st.cache_resource(show_spinner="Memuat model sentimen...")
def load_sentiment_model():
    return pipeline("text-classification", model=MODEL_NAME)


@st.cache_resource
def load_sastrawi():
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover



def clean_and_process_text(text):
    stemmer, stopword_remover = load_sastrawi()
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    cleaned_text = stopword_remover.remove(cleaned_text)
    cleaned_text = stemmer.stem(cleaned_text)
    return cleaned_text


def convert_to_wav(input_path, output_path):
    """Konversi audio/video apa pun (mp3, wav, mp4) ke WAV mono 16 kHz dengan ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Gagal mengonversi file dengan ffmpeg:\n{result.stderr[-1000:]}")
    return output_path


def get_media_duration(path):
    try:
        _, seconds = imageio_ffmpeg.count_frames_and_secs(path)
        return seconds
    except Exception:
        return None


def voice_to_text(wav_path):
    """Speech-to-text dipotong per CHUNK_SECONDS agar audio panjang tidak ditolak Google."""
    recognizer = sr.Recognizer()
    texts = []
    with sr.AudioFile(wav_path) as source:
        total = source.DURATION
        offset = 0.0
        progress = st.progress(0.0, text="Mengubah suara menjadi teks...")
        while offset < total:
            audio_data = recognizer.record(source, duration=CHUNK_SECONDS)
            try:
                texts.append(recognizer.recognize_google(audio_data, language="id-ID"))
            except sr.UnknownValueError:
                pass  # bagian ini tidak terdeteksi ucapan, lanjut ke potongan berikutnya
            except sr.RequestError as e:
                st.error(f"Tidak dapat menghubungi Google Speech Recognition: {e}")
                break
            offset += CHUNK_SECONDS
            progress.progress(min(offset / total, 1.0) if total else 1.0,
                              text="Mengubah suara menjadi teks...")
        progress.empty()

    text = " ".join(texts).strip()
    if not text:
        st.error("Google Speech Recognition tidak dapat mengenali ucapan pada audio.")
    return text


# ---------------------------------------------------------------------------
# Analisis sentimen
# ---------------------------------------------------------------------------
def nlp_processing(text):
    nlp = load_sentiment_model()
    # truncation agar teks panjang (>512 token) tidak membuat model error
    return nlp(text, truncation=True, max_length=512)[0]


def map_sentiment_category(label):
    """Petakan label model ke skor -1 / 0 / 1.

    Catatan: versi lama memakai `score` (tingkat keyakinan model), bukan label,
    sehingga hasil sentimen salah. Sekarang memakai label prediksi.
    """
    label = label.lower()
    if "pos" in label or label == "label_0":
        return 1
    if "neg" in label or label == "label_2":
        return -1
    return 0


SENTIMENT_NAME = {1: "Positif", 0: "Netral", -1: "Negatif"}

def count_words(text):
    word_freq = {}
    for word in text.split():
        word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq


def create_tag_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def create_word_frequency_chart(word_freq):
    top = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:10]
    top_words = [w for w, _ in top]
    top_freqs = [f for _, f in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_words, top_freqs, color="skyblue")
    ax.set_xlabel("Kata")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Top 10 Kata yang Mempengaruhi Sentimen")
    plt.setp(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    plt.close(fig)


def display_result_with_tag_cloud(text, nlp_result, category):
    sentiment_score = map_sentiment_category(nlp_result["label"])
    sentiment_label = SENTIMENT_NAME[sentiment_score]

    st.subheader(category)
    st.write(f"**Analisis Sentimen:** {sentiment_score} - {sentiment_label} "
             f"(label model: {nlp_result['label']}, keyakinan: {nlp_result['score']:.2%})")

    if not text.strip():
        st.warning("Teks kosong setelah dibersihkan, tag cloud tidak dapat dibuat.")
        return

    create_tag_cloud(text)
    create_word_frequency_chart(count_words(text))

def analyze_wav(wav_path, title):
    file_text = voice_to_text(wav_path)
    if not file_text:
        return

    st.write("**Voice to Text result:**")
    st.write(file_text)

    cleaned_text = clean_and_process_text(file_text)
    st.write("**Cleansing Data result:**")
    st.write(cleaned_text if cleaned_text else "_(kosong)_")

    with st.spinner("Menganalisis sentimen..."):
        # analisis pada teks asli bila hasil cleansing kosong
        nlp_result = nlp_processing(cleaned_text or file_text)

    display_result_with_tag_cloud(cleaned_text, nlp_result, title)


def handle_upload(uploaded_file, is_video):
    with tempfile.TemporaryDirectory() as tmpdir:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        input_path = os.path.join(tmpdir, f"input{ext}")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' berhasil diunggah!")

        if is_video:
            st.video(input_path)
        else:
            st.audio(input_path)

        duration = get_media_duration(input_path)
        if duration:
            st.write(f"Durasi: {duration:.1f} detik")

        wav_path = os.path.join(tmpdir, "audio.wav")
        try:
            with st.spinner("Mengekstrak / mengonversi audio..."):
                convert_to_wav(input_path, wav_path)
        except RuntimeError as e:
            st.error(str(e))
            return

        prefix = "Video" if is_video else "Audio"
        analyze_wav(wav_path, f"Hasil dari {prefix} {uploaded_file.name}")


def main():
    st.title("Analisis Sentimen Video & Audio")
    menu = st.sidebar.selectbox("Pilih Menu", ["Upload Audio", "Upload Video"])

    if menu == "Upload Audio":
        uploaded = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
        if uploaded is not None:
            handle_upload(uploaded, is_video=False)
    else:
        uploaded = st.file_uploader("Upload Video File", type=["mp4"])
        if uploaded is not None:
            handle_upload(uploaded, is_video=True)


if __name__ == "__main__":
    main()
