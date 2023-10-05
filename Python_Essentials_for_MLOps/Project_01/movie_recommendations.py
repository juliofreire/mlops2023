"""
Interactive Movie Recommendation System with TF-IDF and User Ratings Analysis
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# https://files.grouplens.org/datasets/movielens/ml-25m.zip
# Using Pandas to open a csv file
movies = pd.read_csv('Python_Essentials_for_MLOps/Project_01/movies.csv')

# Take a glimpse of the data
movies.head()

def clean_title(title):
    """Clean up titles by removing special characters"""
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

# Apply the cleaning on the dataset and save in another column
movies["clean_title"] = movies["title"].apply(clean_title)
print(movies)


def search(title):
    """
    Search for movie titles that are similar to the provided title using TF-IDF vectorization.
    
    Args:
        title (str): The movie title to search for.

    Returns:
        pd.DataFrame: A DataFrame containing the top 5 movie titles that are most similar to the 
    input title.
        
    Description:
        This function takes a movie title as input and uses TF-IDF (Term Frequency-Inverse 
    Document Frequency) vectorization to find similar movie titles from a pre-existing dataset.
    It performs the following steps:
    
    1. Creates a TF-IDF Vectorizer that supports unigrams and bigrams.
    2. Transforms a collection of cleaned movie titles into a TF-IDF matrix.
    3. Cleans the input title using a 'clean_title' function (not shown) to prepare it 
    for comparison.
    4. Converts the cleaned input title into a TF-IDF vector.
    5. Calculates the cosine similarity between the input title's vector and the TF-IDF matrix.
    6. Identifies the indices of the top 5 most similar movie titles.
    7. Returns a DataFrame containing these similar movie titles in descending order of similarity.
    
    This function is useful for finding movies with titles similar to a given query title, which can
    be valuable for recommendation systems or searching large movie databases.
    """

    # Create a TF-IDF Vectorizer with unigram and bigram support
    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Transform the clean movie titles into a TF-IDF matrix
    tfidf = vectorizer.fit_transform(movies["clean_title"])

    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results


MOVIE_ID = 89745


def find_similar_movies(movie_id):
    """
    Find and recommend movies similar to a given movie based on user ratings.
    
    Args:
        movie_id (int): The ID of the movie for which similar movies are to be found.

    Returns:
        pd.DataFrame: A DataFrame containing recommended movies along with their scores,
    titles, and genres.

    Description:
        This function takes a movie ID as input and uses user ratings from a dataset to recommend 
    movies that are similar in terms of user preferences. It follows these steps:

    1. Loads a ratings dataset (assumed to be stored in a CSV file).
    2. Identifies users who have rated the given movie highly (ratings greater than 4).
    3. Collects movies that these similar users have rated highly.
    4. Calculates the percentage of similar users who rated each of these movies.
    5. Filters movies that received a significant percentage of 
    high ratings (greater than 10% of similar users).
    6. Calculates the overall popularity of these filtered movies among all users.
    7. Computes a recommendation score for each movie based on the ratio of similar
    user ratings to overall ratings.
    8. Sorts the movies by score in descending order.
    9. Returns the top 10 recommended movies, including their scores, titles, and genres.

    This function is designed to provide movie recommendations by leveraging user ratings, 
    making it useful in recommendation systems and movie recommendation applications.
    """
    ratings = pd.read_csv("Python_Essentials_for_MLOps/Project_01/ratings.csv")
    similar_users = ratings[(ratings["movieId"] == movie_id) &
                            (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                                (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                        (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True,
                                        right_on="movieId")[["score", "title", "genres"]]


# Create a text input widget for entering a movie title.
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

recommendation_list = widgets.Output()

def on_type(data):
    """
    Handle user input and dynamically update movie recommendations based on the typed text.

    Args:
        data (dict): A dictionary containing user input data.

    Description:
        This function is designed to be used in an interactive environment (like Jupyter Notebook) 
    with widgets. It responds to user input events, such as typing, and performs the 
    following actions:

    1. Clears the existing content within the 'recommendation_list' output widget.
    2. Extracts the typed text from the user input.
    3. Checks if the length of the typed text is greater than 5 characters.
    4. If the text is sufficiently long, it uses the 'search' function to find movie recommendations 
    based on the input text.
    5. Extracts the movie ID of the top recommendation.
    6. Displays a list of similar movies based on the top recommendation using the 
    'find_similar_movies' function.

    This function is typically used in conjunction with widgets to create an interactive movie
    recommendation system that updates in real-time as the user types or provides input.
    """
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
