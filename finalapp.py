from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import requests

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://next-movie-recommender.vercel.app"])

import os
from dotenv import load_dotenv
load_dotenv()
tmdb_key = os.environ['TMDB_ACCESS_KEY']

import pymongo
client = pymongo.MongoClient("mongodb+srv://" + os.environ["MONGO_USER"]+ ":"+ os.environ["MONGO_PWD"]+"@cluster0.tnymq.mongodb.net/?retryWrites=true&w=majority")
db = client['recommender']
collection = db['movie_recommendation_tmdb']

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

        recommended_movies = collection.find_one({"tmdbId": tmdbId})
        if recommended_movies:
          recommended_movies = recommended_movies["recommended_movies"][start_index:end_index]
        else:
          return jsonify({"error": "The data for this movie is currently not available. Please try again later or reach out for further assistance."}), 500

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

        pipeline = [
            {
                "$search": {
                    "autocomplete": {
                        "query": userinput,
                        "path": "title",
                        # "fuzzy": {
                        #     "maxEdits": 2
                        # }
                    }
                },
            },
            {        "$limit": 10  },
            {
                "$project": {
                    "title": 1,
                    "tmdbId": 1,
                    "_id": 0
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        return jsonify(results), 200
    except Exception as e:
        print(e)
        return jsonify([]), 500

@app.route('/id/')
def id():
    try:
        tmdbId = request.args.get("tmdbId", type=int)
        if not tmdbId:
            return jsonify({}), 400

        title = collection.find_one({"tmdbId": tmdbId},{"_id":0, "title":1})

        return jsonify(title), 200
    except Exception as e:
        print(e)
        return jsonify({}), 500

@app.errorhandler(404)
def invalid_route(e): 
    return jsonify({'errorCode' : 404, 'message' : 'Route not found'}), 404

if __name__ == '__main__':
      app.run(host='127.0.0.1')
      
      