# Movie Recommendation Backend

This is a movie recommendation backend system built using Python, Flask, Pandas, Sci-kit Learn and MongoDB. The system is designed to help movie lovers discover new films that match their tastes by combining both collaborative and content-based filtering.

## Collaborative Filtering

The system uses collaborative filtering to generate recommendations based on user ratings. The data is sourced from the [MovieLens 25M movie ratings dataset](https://grouplens.org/datasets/movielens/25m/ 'MovieLens 25M movie ratings dataset').

## Content-Based Filtering

In addition to collaborative filtering, the system also implements content-based filtering by considering the movie genres. This provides a more comprehensive and diverse set of recommendations for users.

## Optimization

The initial version of the system was implemented using the app.py file, which ran the machine learning algorithm in real-time. However, this was found to be slow and inefficient. To resolve this issue, custom scripts were created that pre-processed the inputs and outputs and fed them to MongoDB. This improved the processing time significantly.

The final version of the system, finalapp.py, is the current implementation. This version runs the API on Flask and sources the data from MongoDB, which contains pre-processed inputs and outputs. This eliminates the need to read from CSV files and process the data in real-time at a 10x faster speed.

## Search Autocomplete

The system also includes a search autocomplete feature, which is now implemented using MongoDB's index search feature. This eliminates the need to read from CSV files and provides a faster and more efficient search experience for users.

## Running the System

To run the system locally, simply clone the project and run the finalapp.py file. However, you will need to source the data from the [MovieLens 25M movie ratings dataset](https://grouplens.org/datasets/movielens/25m/ 'MovieLens 25M movie ratings dataset') for the CSV files as the app.py file still needs them.

_Please note that this project is for educational and demonstration purposes only and should not be used in a production environment._
