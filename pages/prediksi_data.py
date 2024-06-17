import re
import os
import joblib
import spacy
import time
import streamlit as st
import pandas as pd

from streamlit_navigation_bar import st_navbar
from hydralit_components import HyLoader, Loaders
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(
    page_title="Sentimen Komentar Instagram",
    layout="wide",
    initial_sidebar_state="collapsed"
)

navbar = st_navbar(["Beranda", "Prediksi Data", "Prediksi Komentar", "Dataset"], selected="Prediksi Data")

if navbar == "Beranda":
    st.switch_page("beranda.py")
elif navbar == "Prediksi Komentar":
    st.switch_page("pages/prediksi_komentar.py")
elif navbar == "Dataset":
    st.switch_page("pages/dataset.py")

@st.cache_data
def load_pipeline(pipeline):
    return joblib.load(pipeline)

if os.path.exists("text_clf_lsvc_baru.joblib"):
    text_clf_lsvc = load_pipeline("text_clf_lsvc_baru.joblib")
else:
    text_clf_lsvc = load_pipeline("text_clf_lsvc.joblib")

# Text Processing
def case_folding(text):
    return text.lower()

def cleaning(text):
    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9]+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)
    text = text.strip()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text

load_dotenv(".env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=3600)
def translate(text_list):
    translated_text = ""
    success = False

    while not success:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": f"You are an expert translator. Translate the following sentence {text_list} from Indonesian to formal English. Just return the translated sentence without any explanation or additional text. IMPORTANT: No explanation or additional text besides the translated sentence itself!!!"
                }
            ],
            temperature=1,
            max_tokens=10_000,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            translated_text += chunk.choices[0].delta.content or ""
        
        success = True

    return translated_text

def translate_in_batches(df, column_name, batch_size=10):
    all_translated_texts = []

    for i in range(0, len(df), batch_size):
        batch = list(df[column_name][i:i + batch_size])
        translated_batch = [translate(text) for text in batch]
        all_translated_texts.extend(translated_batch)

    return all_translated_texts

def clean_string(s):
    s = s.strip()

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1].replace('"', '')

    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1].replace("'", '')

    return s

def clean_list(lst):
    return [clean_string(s) for s in lst]

def lemmatization(text, allowed_postags=["NOUN", "ADJ", "VERB", "ADV", "PROPN"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    doc = nlp(text)
    new_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return " ".join(new_text)

def remove_stopwords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    new_text = [token.text for token in doc if not token.is_stop]
    return " ".join(new_text)

st.subheader("Unggah Data Baru (.csv)")
col1, col2 = st.columns([1, 2.5])
with col1:
    uploaded_file = st.file_uploader("label", type="csv", label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_pred = None

    if df.columns.str.lower().tolist() == ["username", "comment"]:
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

        col1, col2, col3 = st.columns([1, 5, 0.5])
        with col1:
            st.subheader("Hasil Prediksi")

        if "df_pred" not in st.session_state:
            with HyLoader("", loader_name=Loaders.pulse_bars):
                df_pred = df.copy()
                df_pred["case_folding"] = df_pred["comment"].apply(case_folding)
                df_pred["cleaning"] = df_pred["case_folding"].apply(cleaning)
                df_pred = df_pred[df_pred["cleaning"] != ""].reset_index(drop=True)

                all_translated_texts = translate_in_batches(df_pred, "cleaning", batch_size=10)
                all_translated_texts = clean_list(all_translated_texts)

                df_pred["translated_comment"] = all_translated_texts
                df_pred = df_pred[["username", "comment", "translated_comment"]]
                df_pred["case_folding"] = df_pred["translated_comment"].apply(case_folding)
                df_pred["cleaning"] = df_pred["case_folding"].apply(cleaning)
                df_pred = df_pred[df_pred["cleaning"] != ""].reset_index(drop=True)
                df_pred["lemmatization"] = df_pred["cleaning"].apply(lemmatization)
                df_pred["remove_stopwords"] = df_pred["lemmatization"].apply(remove_stopwords)
                df_pred["sentimen"] = text_clf_lsvc.predict(df_pred["remove_stopwords"])
                df_pred = df_pred[["username", "comment", "sentimen"]]

                st.session_state.df_pred = df_pred

        st.dataframe(st.session_state.df_pred, use_container_width=True, hide_index=True)

        if not st.session_state.df_pred.empty:
            with col3:
                unduh = st.download_button(
                    label="Unduh",
                    data=st.session_state.df_pred.to_csv(index=False),
                    file_name="Hasil Prediksi.csv",
                    type="primary",
                    use_container_width=True,
                    mime="text/csv"
                )

    else:
        error = st.error('Format data tidak sesuai. Pastikan data memiliki kolom "Username" dan "Comment".', icon="🚨")
        time.sleep(5)
        error.empty()