# Imported the required libraries.

import pandas as pd
import time
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

# Loaded ratings, movie_metadata, and links datasets.

ratings = pd.read_csv("archive/ratings.csv")
movies = pd.read_csv("archive/movies_metadata.csv", low_memory=False)
links = pd.read_csv("archive/links.csv")

# Converted ID columns to numeric so that merging anfd filtering works properly.
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")

# Merged links with movie metadata to get movieId.
movie_titles = (links.merge(movies[["id", "title"]],left_on="tmdbId",right_on="id",how="left")[["movieId", "title"]])

movie_titles["movieId"] = pd.to_numeric(movie_titles["movieId"], errors="coerce")

# Filtered active users and movies

# Kept users and movies with atleast 20 ratings.
min_user_rating = 20
min_item_rating = 20

# Counting how many ratings each user has made.
user_counts = ratings["userId"].value_counts()

# Counting how many ratings each movie has received.
item_counts = ratings["movieId"].value_counts()

# Kept ratings only from active users and active movies, then index was reset.
ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_user_rating].index) & ratings["movieId"].isin(item_counts[item_counts >= min_item_rating].index)].reset_index(drop=True)

# Converted ratings into a Surprise dataset and split them into training and testing sets.
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[["userId","movieId","rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Created the SVD model and set its training parameters.
svd = SVD(n_factors=80,reg_all=0.02,lr_all=0.005,n_epochs=5,random_state=42)

# Started training the SVD model and recorded the start and end time.
print("Training SVD model")
start = time.time()
svd.fit(trainset)
end = time.time()

# Evaluated the trained model on the test set and printed RMSE and MAE.
print(f"Training completed in {(end - start):.2f} seconds")
print("\nEvaluation:")
predictions = svd.test(testset)
rmse(predictions)
mae(predictions)

# 5. Saved the trained model using pickle
with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)

# Saved the cleaned movie title to a CSV file.
movie_titles.to_csv("movie_titles_clean.csv", index=False)

print("\nModel saved as svd_model.pkl")
print("Movie titles saved as movie_titles_clean.csv")