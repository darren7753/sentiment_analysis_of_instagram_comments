import re
import os
import spacy
import joblib
import pandas as pd

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

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

if os.path.exists("Data/data_latih_lama.csv"):
    original_data = pd.read_csv("Data/data_latih_lama.csv")
    original_data.columns = original_data.columns.str.lower()
else:
    original_data = pd.read_csv("Data/final_data.csv")
    original_data.columns = original_data.columns.str.lower()

new_data = pd.read_csv("Data/data_latih_baru.csv")
new_data.columns = new_data.columns.str.lower()

new_data["case_folding"] = new_data["comment"].apply(case_folding)
new_data["cleaning"] = new_data["case_folding"].apply(cleaning)
new_data = new_data[new_data["cleaning"] != ""].reset_index(drop=True)

all_translated_texts = translate_in_batches(new_data, "cleaning", batch_size=10)
all_translated_texts = clean_list(all_translated_texts)

new_data["translated_comment"] = all_translated_texts
new_data = new_data[["username", "sentimen", "comment", "translated_comment"]]
new_data["case_folding"] = new_data["translated_comment"].apply(case_folding)
new_data["cleaning"] = new_data["case_folding"].apply(cleaning)
new_data = new_data[new_data["cleaning"] != ""].reset_index(drop=True)
new_data["lemmatization"] = new_data["cleaning"].apply(lemmatization)
new_data["remove_stopwords"] = new_data["lemmatization"].apply(remove_stopwords)

df = pd.concat([original_data, new_data], ignore_index=True)
df = df.dropna().reset_index(drop=True)

X = df["remove_stopwords"]
y = df["sentimen"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

text_clf_lsvc = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("smoteenn", SMOTEENN(random_state=42)),
    ("clf", SVC(kernel="linear", class_weight="balanced", random_state=42)),
])

text_clf_lsvc.fit(X_train, y_train)

df.to_csv("Data/data_latih_lama.csv", index=False)
joblib.dump(text_clf_lsvc, "text_clf_lsvc_baru.joblib")