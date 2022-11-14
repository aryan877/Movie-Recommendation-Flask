from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

@app.route('/recommend/<int:tmdbId>')
def recommendmovie(tmdbId):

   # read relevant csv files
    linkscsv = pd.read_csv('./links.csv',encoding= 'unicode_escape')
    moviedatacsv = pd.read_csv('./moviedata.csv', encoding='unicode_escape')
    ratingdatacsv = pd.read_csv('./ratingdata.csv',encoding= 'unicode_escape', nrows=500000)

    # Extract movieId value by using tmbdId, will fail if tmdbId does not exist
    movieId= linkscsv.loc[linkscsv['tmdbId'] == tmdbId, 'movieId'].iloc[0]
 
    #1. Collaborative Filtering ( Item Based )

    # create a matrix with the first 200000 rows in ratingdata with userId as rows and movieId as columns
    movie_matrix = ratingdatacsv.pivot_table(index='userId', columns='movieId', values='rating')

    # create a dataframe that contains the count of ratings for a particular movie 
    rating_count_dataframe = pd.DataFrame(ratingdatacsv.groupby('movieId')['rating'].count()).rename(columns={'rating': 'number_of_ratings'})
    # Extract input movie ratings by all users, will fail if queried movie was not amongst the first 200000 movies
    queried_movie_ratings_column = movie_matrix[movieId]

    # create a dataframe that shows correlation of input movie rating column with all other movie columns in the matrix
    correlation_dataframe = movie_matrix.corrwith(queried_movie_ratings_column, axis = 0)

    # rename the column as Correlation
    correlation_dataframe = pd.DataFrame(correlation_dataframe, columns=['Correlation'])

    # Drop movies that have no correlation
    correlation_dataframe.dropna(inplace=True)

    # rename the index as movieId, corrwith took the movieId from matrix as the index in the 3rd last step
    correlation_dataframe = correlation_dataframe.rename_axis('movieId')

    # merge rating count with the result of correlation in order to drop low rating count movies
    correlation_dataframe = pd.merge(correlation_dataframe, rating_count_dataframe, on='movieId', how='left')

    # drop movies with less than 5 ratings
    correlation_dataframe = correlation_dataframe[correlation_dataframe['number_of_ratings'] > 10]

    # drop movies with less than 0.5 correlation and then sort the results in the same step
    correlation_dataframe = correlation_dataframe[correlation_dataframe['Correlation'] > 0.5].sort_values(by='Correlation', ascending=False)
    
    # merge genres of the movies to the correlation dataframe
    correlation_dataframe = correlation_dataframe.merge(moviedatacsv[['genres','movieId']], on='movieId')

    # merge tmdbId of the movies to the correlation dataframe   
    correlation_dataframe = correlation_dataframe.merge(linkscsv[['tmdbId','movieId']], on='movieId')

    # format the genres string for further processing
    correlation_dataframe['genres'] = correlation_dataframe['genres'].str.lower()
    correlation_dataframe['genres'] = correlation_dataframe['genres'].str.replace('|',' ')

    # set index of correlation dataframe as tmdbId
    correlation_dataframe = correlation_dataframe.set_index('tmdbId') 
   
    #2. Content Based Filtering
    count = CountVectorizer()

    # out of all the filtered results post collaborative filtering perform a count vectorization on the genres
    count_matrix = count.fit_transform(correlation_dataframe['genres'])

    # create a cosine similarity matrix using the count vectorization results
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # initialize result array to be returned to the caller
    recommended_movies = []

    # indices are the tmbdId's, create a Panda Series out of index
    indices = pd.Series(correlation_dataframe.index)

    # getting index corresponding to the tmdbId
    idx = indices[indices == tmdbId].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # get the top results from the series in the form of indexes
    top_results = list(score_series.iloc[0:36].index)

    # iteratively append to the result array
    for i in top_results:

        # get tmdbId corresponding to i 
        val = correlation_dataframe.index[i]

        # do not add the queried movie at any iteration
        if tmdbId != val:
            recommended_movies.append(val)

    # add the queried movie to the result array at the start
    recommended_movies.insert(0, float(tmdbId))

    return jsonify(recommended_movies)


#this method will receive the name of the movie ( string )
@app.route('/namefilter/<string:userinput>')
def namefilter(userinput):

    # read relevant csv files 
    moviedatacsv = pd.read_csv('./moviedata.csv', encoding='unicode_escape')
    linkscsv = pd.read_csv('./links.csv', encoding='unicode_escape')

    # filter names of movies based on query string
    filterednames = moviedatacsv[moviedatacsv['title'].str.contains(userinput, flags = re.IGNORECASE)].reset_index(drop=True)
   
    # discard except top 10
    filterednames = filterednames.head(10)

    # merge with movieId
    filterednameswithtmdbId = pd.merge(filterednames, linkscsv, on='movieId', how='left')

    # send movieId along with movieName as json
    return filterednameswithtmdbId.to_json(orient='records'), 200


@app.errorhandler(404)
def invalid_route(e): 
    return jsonify({'errorCode' : 404, 'message' : 'Route not found'}), 404


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
      
      