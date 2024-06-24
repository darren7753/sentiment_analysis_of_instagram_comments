# Sentiment Analysis of Instagram Comments <a href="https://sentimen-komen-instagram.streamlit.app/" target="_blank"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a>

This project was commissioned by a client on 2024-06-15. If you're interested in similar work, check out my freelance data analyst profile on [Fastwork](https://fastwork.id/user/darren7753).

## Objective
This project aims to perform sentiment analysis using a linear kernel SVM on an Instagram comment dataset provided by the client. Additionally, a Streamlit app was developed for seamless model deployment and user interaction.

## Sentiment Analysis
The first step in the process was text preprocessing, which included a range of tasks from removing unnecessary characters to lemmatization. A significant challenge was translating Indonesian slang into formal English, as most text processing libraries, such as SpaCy, are optimized for English. To address this, I utilized Meta's 70B parameter LLaMA 3 model for translation, which proved more effective than tools like Google Translate. Below is a sample of 3 rows from the dataset after preprocessing:
<table>
  <thead>
    <tr>
      <th>username</th>
      <th>sentimen</th>
      <th>comment</th>
      <th>translated_comment</th>
      <th>case_folding</th>
      <th>cleaning</th>
      <th>lemmatization</th>
      <th>remove_stopwords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>pkk_desakisik	</td>
      <td>positif</td>
      <td>terima kasih Bu Yani beserta Rombongan sudah datang di Desa Kisik</td>
      <td>We would like to thank Mrs. Yani and her entourage for visiting Kisik Village.</td>
      <td>we would like to thank mrs. yani and her entourage for visiting kisik village.</td>
      <td>we would like to thank mrs  yani and her entourage for visiting kisik village</td>
      <td>like thank mrs yani entourage visit kisik village</td>
      <td>like thank mrs yani entourage visit kisik village</td>
    </tr>
    <tr>
      <td>abde_prastio</td>
      <td>positif</td>
      <td>alhamdulillah makin keren kabupatenku sekarang 😍</td>
      <td>Praise be to God, my regency is amazing now.</td>
      <td>praise be to god, my regency is amazing now.</td>
      <td>praise be to god  my regency is amazing now</td>
      <td>praise god regency amazing now</td>
      <td>praise god regency amazing</td>
    </tr>
    <tr>
      <td>maarif1515</td>
      <td>positif</td>
      <td>Jalan poros kabupaten yang menghubungkan dari desa dampaan sampai dungus mohon untuk di tinjau</td>
      <td>The highway connecting from Dampaan Village to Dungus, which passes through the district's axis, is requested to be reviewed.</td>
      <td>the highway connecting from dampaan village to dungus, which passes through the district's axis, is requested to be reviewed.</td>
      <td>the highway connecting from dampaan village to dungus  which passes through the district s axis  is requested to be reviewed</td>
      <td>highway connect dampaan village dungus pass district axis request review</td>
      <td>highway connect dampaan village dungus pass district axis request review</td>
    </tr>
  </tbody>
</table>

The dataset was then split into training and testing sets with an 80:20 ratio. The pipeline used consisted of:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts text into numerical features by evaluating the importance of words within the context of the entire dataset.
- **SMOTEENN (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors)**: Addresses class imbalance by oversampling minority classes and removing noise from the data.
- **Linear Kernel SVM (Support Vector Machine)**: A machine learning model that finds the optimal hyperplane for classifying data, effective for linearly separable data.

The model achieved an accuracy score of 78.24%.

## Streamlit Web App
The Streamlit web app consists of 4 pages: Beranda (Home), Prediksi Data (Data Prediction), Prediksi Komentar (Comment Prediction), and Dataset, each offering unique features.

### Beranda (Home)
<img alt="Beranda (Home) Page" src="https://github.com/darren7753/sentiment_analysis_of_instagram_comments/assets/101574668/8a287d45-0100-4e02-a8a6-44c149e724ed">

This page serves as an introduction. It provides an overview of the web app on the left side and displays a pie chart of sentiment distribution in the dataset on the right side.

### Prediksi Data (Data Prediction)
https://github.com/darren7753/sentiment_analysis_of_instagram_comments/assets/101574668/775234c4-f1d5-41e9-90c3-3afe69c32717

This page allows users to upload a new dataset. The trained model will then predict the sentiments of the uploaded data.

### Prediksi Komentar (Comment Prediction)
https://github.com/darren7753/sentiment_analysis_of_instagram_comments/assets/101574668/c40261ca-59cc-46b8-bd2f-3564f0f74cec

This page allows users to type in text. The trained model will then predict the sentiment of the entered text.

### Dataset
https://github.com/darren7753/sentiment_analysis_of_instagram_comments/assets/101574668/0ca48fad-cb7d-469c-a1a9-879ff66a59ad

This page allows users to upload a new training dataset. Once the dataset is uploaded, the model will automatically retrain and update based on the new data.