"""This code provide a movie recomendation based on what films someone watched."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# https://files.grouplens.org/datasets/movielens/ml-25m.zip
# Using pandas to open
movies = pd.read_csv('Python_Essentials_for_MLOps/Project_01/movies.csv')

movies.head()

def clean_title(title):
    """ DESCRIPTION"""
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

movies["clean_title"] = movies["title"].apply(clean_title)
print(movies)

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])


def search(title):
    """ DESCRIPTION"""
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results

#pip install ipywidgets
#jupyter labextension install @jupyter-widgets/jupyterlab-manager



movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()


# def on_type(data):
#     """#ODO:description"""
#     with movie_list:
#         movie_list.clear_output()
#         title = data["new"]
#         if len(title) > 5:
#             display(search(title))

# movie_input.observe(on_type, names='value')


display(movie_input, movie_list)
MOVIE_ID = 89745

# def find_similar_movies(movie_id):
# movie = movies[movies["movieId"] == movie_id]


# print(ratings.dtypes)

# similar_users = ratings[(ratings["movieId"] == movie_id) &
#                         (ratings["rating"] > 4)]["userId"].unique()

# similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
#                             (ratings["rating"] > 4)]["movieId"]

# similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

# similar_user_recs = similar_user_recs[similar_user_recs > .10]
# all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
# all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
# rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
# rec_percentages.columns = ["similar", "all"]

# print(rec_percentages)

# rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
# rec_percentages = rec_percentages.sort_values("score", ascending=False)
# rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

def find_similar_movies(movie_id):
    """ DESCRIPTION"""
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


movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

recommendation_list = widgets.Output()

def on_type(data):
    """DESCRIPTION"""
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
