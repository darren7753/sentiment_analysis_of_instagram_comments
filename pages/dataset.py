import os
import time
import streamlit as st
import pandas as pd

from streamlit_navigation_bar import st_navbar
from github import Github, Auth
from dotenv import load_dotenv
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Sentimen Komentar Instagram",
    layout="wide",
    initial_sidebar_state="collapsed"
)

navbar = st_navbar(["Beranda", "Prediksi Data", "Prediksi Komentar", "Dataset"], selected="Dataset")

if navbar == "Beranda":
    st.switch_page("beranda.py")
elif navbar == "Prediksi Data":
    st.switch_page("pages/prediksi_data.py")
elif navbar == "Prediksi Komentar":
    st.switch_page("pages/prediksi_komentar.py")

st.subheader("Unggah Data Latih (.csv)")
col1, col2 = st.columns([1, 2.5])
with col1:
    uploaded_file = st.file_uploader("label", type="csv", label_visibility="collapsed")

load_dotenv(".env")
github_api_token = os.getenv("GITHUB_API_KEY")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    file_path = "Data/data_latih_baru.csv"

    if data.columns.str.lower().tolist() == ["username", "comment", "sentimen"]:
        g = Github(github_api_token)
        repo = g.get_repo("darren7753/sentiment_analysis_of_instagram_comments")

        try:
            contents = repo.get_contents(file_path, ref="main")
            repo.delete_file(
                contents.path,
                f"Automated delete {datetime.now() + timedelta(hours=7)}",
                contents.sha,
                branch="main"
            )
            repo.create_file(
                file_path,
                f"Automated update {datetime.now() + timedelta(hours=7)}",
                data.to_csv(index=False),
                branch="main"
            )
            success_message = "Data latih berhasil ditambahkan."
        except:
            repo.create_file(
                file_path,
                f"Automated update {datetime.now() + timedelta(hours=7)}",
                data.to_csv(index=False),
                branch="main"
            )
            success_message = "Data latih berhasil ditambahkan."

        success = st.success(f"{success_message} Model sedang dilatih ulang. Silakan meninggalkan halaman ini.", icon="✅")
        time.sleep(5)
        success.empty()

    else:
        error = st.error('Format data tidak sesuai. Pastikan data memiliki kolom "Username", "Comment", dan "Sentimen".', icon="🚨")
        time.sleep(5)
        error.empty()
