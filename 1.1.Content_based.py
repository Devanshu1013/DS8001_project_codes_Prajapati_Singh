# Install and Import the required libraries.
import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from nltk.stem.porter import PorterStemmer

# Loading data's (Note: The data files are under archive folder).
credits = pd.read_csv('archive/credits.csv')
keywords = pd.read_csv('archive/keywords.csv')
movies = pd.read_csv('archive/movies_metadata.csv')

# Converting the id's of credits and keywords to string for maintaining consistency across data.
credits['id'] = credits['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)

# Merging credits and keywords on base of id and further that is merged to movie_metadata.
credits = credits.merge(keywords, on='id')
movies = movies.merge(credits, on='id')

# Using only the required columns to avoid extra load and for clear data.
movies = movies[['id','title','overview','genres','keywords','cast','crew']]

# Drop the missing and duplicate values from the data.
movies.dropna(inplace=True)
movies = movies.drop_duplicates(subset=['id']).reset_index(drop=True)

# The genres and keywords has an internal JSON format and convert it into List form for using that data for further process.
def convert(obj):
    lis = []
    for i in ast.literal_eval(obj):
        lis.append(i['name'])
    return lis

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# For the cast field we have to use the main cast (eg: Actor, Actress, {main role}) so this function takes the main actors and reduces the noise from cast.
def convert2(obj):
    cast = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            cast.append(i['name'])
            counter += 1
        else:
            break
    return cast

movies['cast'] = movies['cast'].apply(convert2)

# From the crew list this function will find the director's name as the crew list has many jobs and we only want to extract director.
def director(obj):
    Director = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            Director.append(i['name'])
            break
    return Director

movies['crew'] = movies['crew'].apply(director)

# From the overview split the text into words and From the {genres, keywords, cast, crew } removing the internal spaces.
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Now combining the overview, genres, keywords,cast and crew into tags so that we can perform NLTK and further things from one place.
movies['tags'] = (movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast'] + movies['crew'])

# Final DataFrame
movies_final = movies[['id', 'title', 'tags']]
movies_final['tags'] = movies_final['tags'].apply(lambda x: " ".join(x))
movies_final['tags'] = movies_final['tags'].apply(lambda x: x.lower())

# Now with the use of PorterStemmer we will reduce each word from the "tags" field to its root/stem form so that countVectorizer and TF-IDF can use that to compute the things.
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies_final['tags'] = movies_final['tags'].apply(stem)

# Using Cosine Similarity to create the similarity Matrix.
# Here the data is large enough so we break down this into batches to avoid memory error
def compute_cosine_similarity(vectors, batch_size=500):
    n_samples = vectors.shape[0]
    similarity = np.zeros((n_samples, n_samples), dtype=np.float32)
    print("Total samples: ", n_samples)

    # Row-Wise processing the batches 
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        sim_batch = sk_cosine_similarity(vectors[start:end], vectors)
        similarity[start:end] = sim_batch
    return similarity

# Now Using CountVectorizer we will build a movie recommendation system which uses Bag-of-Words and cosine similarity
# And then Save the vectorizer, vectors, similarity matrix and final movie data using pickle.
# Then we will use stored model for recommend movies instantly.
print("CountVectorizer model")
cv = CountVectorizer(max_features=5000, stop_words='english')
cv_vectors = cv.fit_transform(movies_final['tags']).toarray()
print("CV Matrix Shape:",cv_vectors.shape)
cv_similarity = compute_cosine_similarity(cv_vectors, batch_size=500)

pickle.dump(cv, open('CV_vectorizer.pkl', 'wb'))
pickle.dump(cv_vectors, open('CV_vectors.pkl', 'wb'))
pickle.dump(cv_similarity, open('CV_similarity.pkl', 'wb'))
pickle.dump(movies_final, open('CV_movies.pkl', 'wb'))
print("CountVectorizer models saved.\n")

# Other way is to use TF-IDF (Term Frequency-Inverse Document Frequency)
# Here we bulid TF-IDF vectors for each movies and then compute the cosine similarity.
# Then save the models like vectorizer, vectors, similarity matrix and final movies.
print("TF-IDF model")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_vectors = tfidf.fit_transform(movies_final['tags']).toarray()
print("TF-IDF Matrix Shape:",tfidf_vectors.shape)
tfidf_similarity = compute_cosine_similarity(tfidf_vectors, batch_size=500)

pickle.dump(tfidf, open('TFIDF_vectorizer.pkl', 'wb'))
pickle.dump(tfidf_vectors, open('TFIDF_vectors.pkl', 'wb'))
pickle.dump(tfidf_similarity, open('TFIDF_similarity.pkl', 'wb'))
pickle.dump(movies_final, open('TFIDF_movies.pkl', 'wb'))
print("TF-IDF models saved.\n")


# This .py file only preprocess the data and Save the models of CountVectorizer and TF_IDF
# For the recommender ther is second file named "1.2.Content_based_Final_Rec.py"