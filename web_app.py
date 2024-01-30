import streamlit as st
import mlflow
import pandas as pd
import requests
import os
import wandb

# Set the WandB API key using the environment variable
os.environ['WANDB_API_KEY'] = '227bbb648ba9e700838c573b6cca9237ea59641b'

# Initialize WandB with your project name
wandb.init(project='Movie Recommender')

# Set up MLflow
mlflow.set_tracking_uri("http://34.100.213.14:5000/")
model_name = 'movie_rs_svdpp'
version = "1"

# Load the model from MLflow server
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{version}")

# Load the pickled dataframe
df = pd.read_pickle('moviemeta.pkl')

# Loading the movies into a list
movie_list = list(df['title'].unique())

# Function to recommend the movies
def recomd_engine(uid, movie_list):
    testset = [[uid, movie_name, 4] for movie_name in movie_list]
    global model
    predictions = model.test(testset)
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(by='est', ascending=False)
    top_10_movies = list(pred_df.head(10).iid)
    return top_10_movies

# Function to get posters from TMDB
def tmdb_poster(movies, df):
    id = []
    poster = []
    for i in movies:
        id.append(df[df['title'] == i]['tmdbId'].values[0])
    for i in id:
        url = f"https://api.themoviedb.org/3/movie/{i}?api_key=63d30cc474a218f38d16d816eb717270"
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        poster.append(full_path)
    return poster

# Streamlit page design and prediction
st.set_page_config(layout='wide', page_title='Movie Recommender')
st.title('Movie Recommender System')
uid = st.sidebar.number_input("Enter your user ID", format="%d", step=1, value=0)
if st.sidebar.button('Recommend 🚀'):
    with st.spinner("Fetching...."):
        movies = recomd_engine(uid, movie_list)
        posters = tmdb_poster(movies, df)
        st.subheader("Here are the 10 recommended movies for you..!")
        if posters:
            for i in range(0, 10, 5):
                col1, col2, col3, col4, col5 = st.columns(5, gap='medium')
                for j, col in enumerate([col1, col2, col3, col4, col5]):
                    idx = i + j
                    if idx < 10:
                        with col:
                            st.text(movies[idx])
                            st.image(posters[idx])
    
    # Log user ID and top 10 movie recommendations to WandB
    wandb.log({"user_id": uid, "recommended_movies": movies})
