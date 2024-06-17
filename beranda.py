import os
import streamlit as st
import pandas as pd
import altair as alt

from streamlit_navigation_bar import st_navbar

st.set_page_config(
    page_title="Sentimen Komentar Instagram",
    layout="wide",
    initial_sidebar_state="collapsed"
)

navbar = st_navbar(["Beranda", "Prediksi Data", "Prediksi Komentar", "Dataset"])

if navbar == "Prediksi Data":
    st.switch_page("pages/prediksi_data.py")
elif navbar == "Prediksi Komentar":
    st.switch_page("pages/prediksi_komentar.py")
elif navbar == "Dataset":
    st.switch_page("pages/dataset.py")

@st.cache_data
def load_data(data):
    data = pd.read_csv(data)
    return data

if os.path.exists("Data/data_latih_lama.csv"):
    df = load_data("Data/data_latih_lama.csv")
else:
    df = load_data("Data/final_data.csv")

st.markdown("<h1 style='text-align: center;'>Analisis Sentimen Komentar Instagram</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")

col1, col2 = st.columns([1.5, 1])
with col1:
    st.markdown(
        """
        <p style='font-size:17px; text-align: justify'>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque non quam auctor libero molestie commodo at nec leo. Proin porttitor tempor metus, sit amet volutpat ante tristique non. Maecenas turpis lectus, ullamcorper sed fermentum ut, bibendum sed quam. Mauris ut sapien enim. Nullam leo urna, accumsan ac quam eget, blandit euismod libero. In pretium aliquam risus sit amet tempus. Duis sit amet est molestie, tempor ex non, vehicula erat. Suspendisse potenti. Donec sodales nibh a posuere interdum. Aenean feugiat maximus nunc, id auctor est dapibus sit amet. Vivamus at quam feugiat, pretium nisi vel, tristique velit. Curabitur congue mattis posuere. Fusce blandit, justo sed viverra bibendum, augue elit dictum diam, quis auctor lacus arcu non arcu. Etiam vel ullamcorper velit. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque non quam auctor libero molestie commodo at nec leo. Proin porttitor tempor metus, sit amet volutpat ante tristique non. Maecenas turpis lectus, ullamcorper sed fermentum ut, bibendum sed quam. Mauris ut sapien enim. Nullam leo urna, accumsan ac quam eget, blandit euismod libero. In pretium aliquam risus sit amet tempus. Duis sit amet est molestie, tempor ex non, vehicula erat. Suspendisse potenti.
        </p>
        """,
        unsafe_allow_html=True
    )

with col2:
    with st.container(border=True):
        sentiment_counts = df["sentimen"].value_counts().reset_index()
        sentiment_counts.columns = ["sentimen", 'count']
        sentiment_counts["percentage"] = (sentiment_counts["count"] / sentiment_counts["count"].sum())

        chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
            theta="count",
            color="sentimen:N",
            tooltip=["sentimen", "count", alt.Tooltip("percentage", format=".2%")]
        ).properties(
            width=350,
            height=350,
            title="Total Dataset Berdasarkan Label Sentimen"
        ).configure_title(
            fontSize=22
        )

        st.altair_chart(chart, use_container_width=True)
