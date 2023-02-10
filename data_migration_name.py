import pymongo
from dotenv import load_dotenv
load_dotenv()
import os
client = pymongo.MongoClient("mongodb+srv://" + os.environ["MONGO_USER"]+ ":"+ os.environ["MONGO_PWD"]+"@cluster0.tnymq.mongodb.net/?retryWrites=true&w=majority")
db = client['recommender']
# collection = db['movie_names_tmdbId_list']
import csv
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.simplefilter('ignore')
collection_recommend = db['movie_recommendation_tmdb']
collection_movie_names = db['movie_names_tmdbId_list']
# Use the find method to retrieve all documents
cursor = collection_recommend.find({})
# Loop over the cursor to access each document
for document in cursor:
  try:
      # Specify the filter to find the movieName document using TmdbId
      document = collection_movie_names.find_one({"tmdbId": int(document["tmdbId"])})
      
      # Specify the new key-value pair to add to the document
      new_values = {"$set": {"title": document["title"]}}

      # Update the document
      result = collection_recommend.update_one({"tmdbId": int(document["tmdbId"])}, new_values)

      print(new_values, document["tmdbId"])

      # Check if the update was successful
      if result.modified_count == 1:
          print("Document updated successfully.")
      else:
          print("Document update failed.")
  except Exception as e:
      print('error',e)
      continue


cursor.close()
client.close()