# Before Running this file Make sure that you have successfully runned the 1.1.Content_based_Final.py file 
# And you have the all 8 saved models(4- CountVec & 4-TF-IDF).

import pickle
import numpy as np
import matplotlib.pyplot as plt

# load the save saved models of TD-IDF and CountVectorizer
tfidf_movies = pickle.load(open('TFIDF_movies.pkl', 'rb'))
tfidf_similarity = pickle.load(open('TFIDF_similarity.pkl', 'rb'))
tfidf_vectors = pickle.load(open('TFIDF_vectors.pkl', 'rb'))

cv_movies = pickle.load(open('CV_movies.pkl', 'rb'))
cv_similarity = pickle.load(open('CV_similarity.pkl', 'rb'))
cv_vectors = pickle.load(open('CV_vectors.pkl', 'rb'))

# Now using the Cosine Similarity it will recommend the Top 5 similar movies {you can change the number of the recommendation using the parameter named "top_n"}
# It will do cosine similarity for both CountVectorizer and TF-IDF.
# This recommend function will give us the movie name and the similarity score.
def recommend_cosine(movie_name, movies_df, similarity_matrix, top_n=5):
    if movie_name not in movies_df['title'].values:
        print("Movie not found.")
        return []

    idx = movies_df[movies_df['title'] == movie_name].index[0]
    distances = similarity_matrix[idx]
    sorted_idx = np.argsort(distances)[::-1][1:top_n+1]
    top_movies = [(i, distances[i]) for i in sorted_idx]

    recommended = []
    for i, score in top_movies:
        recommended.append((movies_df.loc[i, "title"], score))

    print("Recommendations for:", movie_name)
    for i, (title, score) in enumerate(recommended, start=1):
        print(f"{i}. {title} (Similarity: {score:.3f})")
    return recommended

# Plotting the Comparission between TF-IDF and CountVectorizer
def rec_plot(movie_name, tfidf_recs, cv_recs):
    if not tfidf_recs and not cv_recs:
        print("No recommendations to plot.")
        return

    labels = list({title for title, _ in (tfidf_recs + cv_recs)})
    tfidf_dict = dict(tfidf_recs)
    cv_dict = dict(cv_recs)
    tfidf_scores = [tfidf_dict.get(title, 0) for title in labels]
    cv_scores = [cv_dict.get(title, 0) for title in labels]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, tfidf_scores, width, label='TF-IDF', color="#0a4dcaeb")  
    plt.bar(x + width/2, cv_scores, width, label='CountVectorizer', color="#e79a0b")  
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("Similarity Score")
    plt.title(f"Top Recommendations for '{movie_name}' (Cosine Similarity)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Enter the correct movie name as it will look into the dataset.
    # eg: Hulk, Iron Man, Jumanji, ......
    # But be aware its case sensitive so if possible look into the dataset(movies_metadata.csv) for the correct name(title) of the movie for which you want recommendation 
    movie_name = input("Enter movie name: ")
    print("\nTF-IDF Recommendations\n")
    tfidf_recs = recommend_cosine(movie_name, tfidf_movies, tfidf_similarity, top_n=5)
    print("\nCountVectorizer Recommendations\n")
    cv_recs = recommend_cosine(movie_name, cv_movies, cv_similarity, top_n=5)

    # Graph
    rec_plot(movie_name, tfidf_recs, cv_recs)


# In the terminal:
# Note: Enter the correct Movie name (look for the movie name into dataset "movies_metadata.csv") 
# After that you will se the top recommendations for both TF-IDF and CountVectorizer
# At last the graph/plot will be opened which shows the comparison of the both Count-Vectorizer and TF-IDF.