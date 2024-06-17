import re
import os
import joblib
import spacy
import time
import streamlit as st

from streamlit_navigation_bar import st_navbar
from hydralit_components import HyLoader, Loaders
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(
    page_title="Sentimen Komentar Instagram",
    layout="wide",
    initial_sidebar_state="collapsed"
)

navbar = st_navbar(["Beranda", "Prediksi Data", "Prediksi Komentar", "Dataset"], selected="Prediksi Komentar")

if navbar == "Beranda":
    st.switch_page("beranda.py")
elif navbar == "Prediksi Data":
    st.switch_page("pages/prediksi_data.py")
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

def clean_string(s):
    s = s.strip()

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1].replace('"', '')

    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1].replace("'", '')

    return s

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

st.subheader("Prediksi Komentar")

col1, col2 = st.columns([6, 1])
with col1:
    text = st.text_input(
        label="label",
        placeholder="Tulis di sini...",
        label_visibility="collapsed"
    )

with col2:
    predict = st.button(
        label="Prediksi",
        type="primary",
        use_container_width=True
    )

output_container = st.empty()
if predict:
    output_container.empty()
    if text:
        with HyLoader("", loader_name=Loaders.pulse_bars):
            text = case_folding(text)
            text = cleaning(text)
            text = translate(text)
            text = clean_string(text)
            text = case_folding(text)
            text = cleaning(text)
            text = lemmatization(text)
            text = remove_stopwords(text)

            sentimen = text_clf_lsvc.predict([text])[0]

        output_container.info(f"Sentimen dari teks di atas adalah **{sentimen.upper()}**.", icon="ℹ️")
    else:
        output_container.error("Tuliskan teks yang ingin diprediksi.", icon="🚨")
        time.sleep(5)
        output_container.empty()