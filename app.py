from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
import requests
from dotenv import load_dotenv
load_dotenv()
import os
app = Flask(__name__)
# app.config['DEBUG'] = True
CORS(app)
tmdb_key = os.environ['TMDB_ACCESS_KEY']

@app.route('/recommendations/')
def recommendation():
    try:
        tmdbId = request.args.get("id", type=int)
        page = request.args.get("p", default=1, type=int)
        per_page = 12
        # read relevant csv files
        if page < 1 or page > 3:
            return jsonify({"error": "Invalid page number. Page number should be between 1 and 3"}), 400
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        linkscsv = pd.read_csv('./links.csv',encoding= 'unicode_escape')
        moviedatacsv = pd.read_csv('./moviedata.csv', encoding='unicode_escape')
        ratingdatacsv = pd.read_csv('./ratingdata.csv',encoding= 'unicode_escape', nrows=500000)

        # Extract movieId value by using tmbdId, will fail if tmdbId does not exist
        movieId= linkscsv.loc[linkscsv['tmdbId'] == tmdbId, 'movieId'].iloc[0]
        
        # check if movieId exists in the dataset
        if not movieId:
            return jsonify({"error": "Invalid tmdbId", "message": "The tmdbId you have entered does not exist in our dataset"}), 404
 
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

        # drop movies with less than 8 ratings
        correlation_dataframe = correlation_dataframe[correlation_dataframe['number_of_ratings'] > 8]

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
        recommended_movies = recommended_movies[start_index:end_index]
        # return the recommended_movies in json format
        # return jsonify({"recommended_movies": recommended_movies})

        is_error = False
        tmdb_data = []
        for d in recommended_movies:
            try:
                response = requests.get(f"https://api.themoviedb.org/3/movie/{d}?api_key={tmdb_key}")
                if response.status_code == 200:
                    tmdb_data.append(response.json())
                else:
                    pass
            except:
                is_error = True
                break
        if is_error:
            return jsonify({"error": "Unable to connect to TMDB API"}), 400
        else:
            try:
                response = requests.get(f"https://api.themoviedb.org/3/movie/{tmdbId}?api_key={tmdb_key}")
                if response.status_code == 200:
                    original_title = response.json()["original_title"]
                else:
                    pass
            except:
                is_error = True
            if is_error:
                return jsonify({"error": "Unable to connect to TMDB API"}), 400
            else:
                return jsonify({
                    "movie_list": tmdb_data,
                    "movie_count": len(tmdb_data),
                    "searched_movie_title": original_title
                })

    except Exception as e:
        print(e)
        return jsonify({"error": "Internal Server Error", "message": "An error has occured, please try again later"}), 500


@app.route('/search/')
def search():
    try:
        userinput = request.args.get("name")
        if not userinput:
            return jsonify([]), 200
        
        # read relevant csv files
        moviedatacsv = pd.read_csv('./moviedata.csv', encoding='unicode_escape')
        linkscsv = pd.read_csv('./links.csv', encoding='unicode_escape')

        escaped_userinput = re.escape(userinput)

        # filter names of movies based on query string and limit the number of results to 10
        filterednames = moviedatacsv[moviedatacsv['title'].str.contains(escaped_userinput, flags = re.IGNORECASE, regex=True)].head(10)

        # merge with movieId
        filterednameswithtmdbId = filterednames.merge(linkscsv[['tmdbId', 'movieId']], on='movieId', how='left')

        # send movieId along with movieName as json
        print(filterednameswithtmdbId.to_json(orient='records'))
        return filterednameswithtmdbId.to_json(orient='records'), 200
    except:
        return jsonify([]), 200


@app.errorhandler(404)
def invalid_route(e): 
    return jsonify({'errorCode' : 404, 'message' : 'Route not found'}), 404

if __name__ == '__main__':
      app.run(host='127.0.0.1')
      
      