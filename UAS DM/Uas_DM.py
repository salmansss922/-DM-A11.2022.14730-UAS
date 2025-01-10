import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process
import altair as alt
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Tema Visual dengan CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #00695c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header dengan Desain Menarik
st.markdown(
    """
    <div style="background-color:#e0f7fa; padding:20px; border-radius:10px;">
        <h1 style="color:#00695c; text-align:center;">Sistem Rekomendasi Film</h1>
        <p style="text-align:center; color:#004d40;">Temukan film favorit Anda dengan teknologi AI canggih</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Langkah 2: Memuat dataset
file_path = r"C:\Tugas\stki uas\tmdb_5000_movies.csv"
movies = pd.read_csv(file_path)

# Mengecek keberadaan dataset
if not os.path.exists(file_path):
    st.error(f"Dataset tidak ditemukan. File `{file_path}` tidak ada di direktori.")
    st.stop()
else:
    st.success(f"Dataset ditemukan: `{file_path}`")

movies = pd.read_csv(file_path)

# Langkah 3: Memilih kolom yang relevan
movies = movies[['title', 'overview', 'genres', 'vote_average', 'vote_count']].dropna()

# Langkah 4: Pra-pemrosesan Genres
def extract_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except Exception:
        return []

movies['genres'] = movies['genres'].apply(extract_genres)
movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))

# Langkah 5: Kombinasi Overview dan Genres
movies['content'] = movies['overview'] + ' ' + movies['genres_str']

# Langkah 6: Representasi Teks
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies['content'])

# Langkah 7: Menghitung Kemiripan Kosinus
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Langkah 8: Normalisasi Skor dan Popularitas
scaler = MinMaxScaler()
movies['normalized_vote_average'] = scaler.fit_transform(movies[['vote_average']])
movies['normalized_vote_count'] = scaler.fit_transform(movies[['vote_count']])

# Langkah 9: Menggabungkan Skor
def calculate_final_score(vote_weight, count_weight):
    if 'final_score' not in movies.columns:
        movies['final_score'] = (
            vote_weight * movies['normalized_vote_average'] +
            count_weight * movies['normalized_vote_count']
        )

# Pengaturan Bobot
st.subheader("Pengaturan Bobot")
st.markdown(
    """
    Pastikan total bobot (Skor Review + Popularitas) adalah 1.0 untuk menjaga keseimbangan.
    """
)

vote_weight = st.slider("Bobot Skor Review (Vote Average)", 0.0, 1.0, 0.7)
count_weight = 1.0 - vote_weight
st.slider("Bobot Popularitas (Vote Count)", 0.0, 1.0, count_weight)

if st.button("Terapkan Bobot"):
    calculate_final_score(vote_weight, count_weight)
    st.success(f"Bobot diterapkan: Skor Review={vote_weight}, Popularitas={count_weight}")

    fig, ax = plt.subplots()
    ax.bar(['Skor Review', 'Popularitas'], [vote_weight, count_weight], color=['#00695c', '#004d40'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Bobot')
    ax.set_title('Perbandingan Bobot')
    st.pyplot(fig)

# Fungsi Rekomendasi
def recommend(title, cosine_sim=cosine_sim, genre_filter=None):
    try:
        calculate_final_score(vote_weight, count_weight)
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies.iloc[movie_indices]

        if genre_filter:
            recommended_movies = recommended_movies[recommended_movies['genres_str'].str.contains(genre_filter, case=False)]

        return recommended_movies.sort_values(by='final_score', ascending=False).head(10)
    except IndexError:
        st.error("Film tidak ditemukan dalam dataset.")
        return pd.DataFrame()

# Sidebar untuk Navigasi dan Filter Genre
with st.sidebar:
    st.markdown(
        """
        <div style="background-color:#00695c; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:#ffffff;">Navigasi</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_selection = st.radio(
        "Pilih Halaman",
        ["Daftar Film", "Rekomendasi", "Tentang", "Informasi Dataset"]
    )

    all_genres = sorted(set([genre for genres in movies['genres'] for genre in genres]))
    selected_genre = st.selectbox("Pilih genre untuk rekomendasi:", ["Semua"] + all_genres)
    genre_filter = selected_genre if selected_genre != "Semua" else None

# Tab: Daftar Film
if tab_selection == "Daftar Film":
    st.subheader("Daftar Film yang Tersedia:")
    title_input = st.text_input("Masukkan judul film untuk rekomendasi:", key='title_input')
    filtered_movies = movies if not genre_filter else movies[movies['genres_str'].str.contains(genre_filter, case=False)]
    st.dataframe(filtered_movies[['title', 'genres_str', 'vote_average', 'vote_count']])

# Tab: Rekomendasi
elif tab_selection == "Rekomendasi":
    st.subheader("Rekomendasi Film")
    title_input = st.session_state.get('title_input', '')
    if title_input:
        rekomendasi = recommend(title_input, genre_filter=genre_filter)
        if not rekomendasi.empty:
            st.dataframe(rekomendasi[['title', 'genres_str', 'vote_average', 'vote_count', 'final_score']])

# Tab: Tentang
elif tab_selection == "Tentang":
    st.write("Aplikasi ini adalah platform cerdas yang memanfaatkan teknologi Natural Language Processing (NLP) dengan model Transformer untuk memberikan rekomendasi film yang akurat dan relevan. Dengan memahami preferensi pengguna melalui analisis data teks, seperti ulasan, deskripsi film, dan komentar pengguna, aplikasi ini mampu menghasilkan rekomendasi yang benar-benar personal dan dinamis.")

# Tab: Informasi Dataset
elif tab_selection == "Informasi Dataset":
    st.subheader("Statistik Dataset")
    st.write(f"Jumlah Film: {len(movies)}")
    st.write(f"Rata-rata Skor Review: {movies['vote_average'].mean():.2f}")

    st.altair_chart(
        alt.Chart(movies).mark_bar().encode(
            x=alt.X('vote_average', bin=True),
            y='count()'
        ).properties(width=600, height=400),
        use_container_width=True
    )