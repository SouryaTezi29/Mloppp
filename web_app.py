
import streamlit as st
import mlflow
import pandas as pd
import requests
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="master-shore-411113-41df45dfb65e.json"

#setting the tracking URI
mlflow.set_tracking_uri("http://34.100.213.14:5000/")
model_name = 'movie_rs_svdpp'
version = "1"

#loading the model from mlflow server
model = mlflow.sklearn.load_model(model_uri = f"models:/{model_name}/{version}")

#Loading the pickled dataframe
df = pd.read_pickle('moviemeta.pkl')

#Loading the movies into a list
movie_list = list(df['title'].unique())

#Function to recommend the movies
def recomd_engine(uid,movie_list):
    testset = [[uid,movie_name,4]for movie_name in movie_list]
    global model
    predictions = model.test(testset)
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(by='est',ascending=False)
    top_10_movies = list(pred_df.head(10).iid)
    return top_10_movies

#Function to get posters from TMDB

def tmdb_poster(movies,df):
    id = []
    poster = []
    for i in movies:
        id.append(df[df['title'] == i]['tmdbId'].values[0])
    for i in id:
        url = f"https://api.themoviedb.org/3/movie/{i}?api_key=63d30cc474a218f38d16d816eb717270"
        data = requests.get(url)
        data = data.json()
        print(data)
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
        poster.append(full_path)
    return poster
    
#Streamlit page design and prediction
st.set_page_config(layout='wide',page_title='Movie Recommender')
st.title('Movie Recommender System')
uid = st.sidebar.number_input("Enter your user ID",format="%d",step=1,value=0)
if st.sidebar.button('Recommend 🚀'):
    with st.spinner("Fetching...."):
        movies = recomd_engine(uid,movie_list)
        posters = tmdb_poster(movies,df)
        st.subheader("Here are the 10 recommended movies for you..!")
        if posters:
            col1,col2,col3,col4,col5 = st.columns(5, gap='medium')
            with col1:
                st.text(movies[0])
                st.image(posters[0])
            with col2:
                st.text(movies[1])
                st.image(posters[1])
            with col3:
                st.text(movies[2])
                st.image(posters[2])
            with col4:
                st.text(movies[3])
                st.image(posters[3])
            with col5:
                st.text(movies[4])
                st.image(posters[4])
        if posters:
            col6,col7,col8,col9,col10 = st.columns(5, gap='medium')
            with col6:
                st.text(movies[5])
                st.image(posters[5])
            with col7:
                st.text(movies[6])
                st.image(posters[6])
            with col8:
                st.text(movies[7])
                st.image(posters[7])
            with col9:
                st.text(movies[8])
                st.image(posters[8])
            with col10:
                st.text(movies[9])
                st.image(posters[9])
