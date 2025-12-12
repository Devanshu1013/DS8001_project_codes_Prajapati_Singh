# Imported libraries for recommendations and plotting.

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader
from surprise.accuracy import rmse

# Loaded the trained SVD model from the pickle file.

with open("svd_model.pkl", "rb") as f:
    svd = pickle.load(f)
    
# Loaded the original ratings and movie title mapping.
ratings = pd.read_csv("archive/ratings.csv")
movie_titles = pd.read_csv("movie_titles_clean.csv")

# Defined the recommendation function to generate top-N movies for a given user.

def recommend_full(model, user_id, ratings_df, titles_df, n=5):
    # Checked if the user existed in the ratings dataset.
    if user_id not in ratings_df["userId"].values:
        print(f"User {user_id} not found in ratings dataset.")
        return
    

    # Collected all unique movie IDs from the ratings.
    all_movies = ratings_df["movieId"].unique()
    
    # Found all movies already rated by the given user.
    watched = ratings_df[ratings_df["userId"] == user_id]["movieId"].values

    # Created a list of movies that the user had not seen yet.
    unseen = [m for m in all_movies if m not in watched]

    # Stopped if the user had already rated every movie.
    if not unseen:
        print(f"User {user_id} has rated all movies.")
        return

    # Predicted ratings for all unseen movies using the trained model.
    predictions = [(mid, model.predict(user_id, mid).est) for mid in unseen]
    # Sorted the predictions in descending order of predicted rating.
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Printed the top-N recommended movies with their predicted scores.
    print(f"\nTop {n} recommendations for User {user_id}:\n")
    for mid, score in predictions[:n]:
        title = titles_df.loc[titles_df["movieId"] == mid, "title"]
        title = title.values[0] if not title.empty else "Unknown"
        print(f"{title} → {score:.2f}")

# Defined a function to check RMSE and visualize squared errors on a small sample.
def plot_rmse_graph(model, ratings_df):
    # Took a small random sample of ratings for quick evaluation.
    sample = ratings_df.sample(n=1000, random_state=42)

    # Converted the sampled ratings into a Surprise dataset.
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(sample[["userId", "movieId", "rating"]], reader)

    # Built a trainset and testset from the sample for sample RMSE testing.
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()

    # Generated predictions on the testset using the loaded model.
    predictions = model.test(testset)

    # Calculated squared errors for each prediction.
    errors = [(pred.r_ui - pred.est) ** 2 for pred in predictions]

    # Plotted the squared error distribution for a subset of predictions.
    plt.figure(figsize=(8, 5))
    plt.plot(errors[:300], marker="o", linestyle="", alpha=0.6)
    plt.title("RMSE Error Distribution (Sample of 1,000 Ratings)")
    plt.xlabel("Sample Index")
    plt.ylabel("Squared Error")
    plt.tight_layout()
    plt.show()

    # Printed the RMSE value on the sampled data.
    print("\nFAST RMSE on sample:")
    rmse(predictions)

# This is our main script that plotted the RMSE graph and then asked for a userId to recommend movies.
if __name__ == "__main__":

    # Showeing RMSE graph using the saved SVD model.
    plot_rmse_graph(svd, ratings)

    # Ask for user input
    user_input = input("Enter a userId: ")

    # Tried to convert the input to an integer userId.
    try:
        active_user = int(user_input)
    except ValueError:
        print("Invalid input. Please enter a numeric userId.")
        raise SystemExit

    # Generated and printed top-5 movie recommendations for the active user.
    recommend_full(svd, active_user, ratings, movie_titles, n=5)