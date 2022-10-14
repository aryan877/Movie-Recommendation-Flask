from flask import Flask, jsonify
from flask_cors import CORS
#importing neccesary packages
import pandas as pd 
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
import time

app = Flask(__name__)
CORS(app)

#api for movie click and search
@app.route('/recommend/<int:tmdbId>')
def recommendmovie(tmdbId):
    linkscsv = pd.read_csv('./links.csv',encoding= 'unicode_escape')
    # Extract movieId value by using tmbdId
    try:
        movieId= linkscsv.loc[linkscsv['tmdbId'] == tmdbId, 'movieId'].iloc[0]
    except Exception:
        return jsonify("tmdbId is not valid"),400

    moviedatacsv = pd.read_csv('./moviedata.csv', encoding='unicode_escape')
    ratingdatacsv = pd.read_csv('./ratingdata.csv',encoding= 'unicode_escape', nrows=200000)

    #1. Collaborative Filtering ( Item Based )
    movie_matrix = ratingdatacsv.pivot_table(index='userId', columns='movieId', values='rating')
    number_of_ratings_for_movie = pd.DataFrame(ratingdatacsv.groupby('movieId')['rating'].count()).rename(columns={'rating': 'number_of_ratings'})
    
    try:
        input_movie_all_ratings = movie_matrix[movieId]
    except Exception:
        return jsonify("Oops! Our Database does not have enough data in order to generate recommendations for this movie"), 500

    similar_movie_based_on_input_movie_ratings = movie_matrix.corrwith(input_movie_all_ratings)

    similar_movie_df = pd.DataFrame(similar_movie_based_on_input_movie_ratings, columns=['Correlation'])
    similar_movie_df.dropna(inplace=True)

    
    similar_movie_df = pd.merge(similar_movie_df, number_of_ratings_for_movie, on='movieId', how='left')

    similar_movie_df = similar_movie_df.join(number_of_ratings_for_movie['number_of_ratings'])
    similar_movie_df = similar_movie_df[similar_movie_df['number_of_ratings'] > 5]
    similar_movie_df = similar_movie_df[similar_movie_df['Correlation'] > 0.5].sort_values(by='Correlation', ascending=False)

    # collaborative filtering results are done till here, proceeding for content based filtering on top of these results
    similar_movie_df = similar_movie_df.merge(moviedatacsv[['genres','movieId']], on='movieId')
    similar_movie_df = similar_movie_df.merge(linkscsv[['tmdbId']], on='movieId')
    similar_movie_df['genres'] = similar_movie_df['genres'].str.lower()
    similar_movie_df['genres'] = similar_movie_df['genres'].str.replace('|',' ')
   
    #2. Content Based Filtering
    count = CountVectorizer()
    count_matrix = count.fit_transform(similar_movie_df['genres'])
    # moviename = moviedatacsv[moviedatacsv['title'] == moviename]
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    recommended_movies = []
    similar_movie_df = similar_movie_df.set_index('tmdbId') 
    indices = pd.Series(similar_movie_df.index)
    # indices[:5]
    #getting index of the movie that matches the tmdbId
    idx = indices[indices == tmdbId].index[0]
    #creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    # print(score_series)
    #getting the indexes of the the 10 most similar movies
    top_10_indexes = list(score_series.iloc[0:39].index)
    #populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(similar_movie_df.index)[i])
    if float(tmdbId) in recommended_movies:
        recommended_movies.remove(float(tmdbId))
    recommended_movies.insert(0,float(tmdbId))
    return jsonify(recommended_movies)


#this method will receive the name of the movie ( string )
@app.route('/namefilter/<string:userinput>')
def namefilter(userinput):
    moviedatacsv = pd.read_csv('./moviedata.csv', encoding='unicode_escape')
    linkscsv = pd.read_csv('./links.csv', encoding='unicode_escape')
    filterednames = moviedatacsv[moviedatacsv['title'].str.contains(userinput, flags = re.IGNORECASE)].reset_index(drop=True)
    filterednames = filterednames.head(10)
    filterednameswithtmdbId = pd.merge(filterednames, linkscsv, on='movieId', how='left')
    return filterednameswithtmdbId.to_json(orient='records'), 200


@app.route('/namefilter/')
def namefilterempty():
    return jsonify([]), 200


@app.errorhandler(404)
def invalid_route(e): 
    return jsonify({'errorCode' : 404, 'message' : 'Route not found'}), 404


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
      
      