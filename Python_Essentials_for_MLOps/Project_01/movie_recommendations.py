"""
Interactive Movie Recommendation System with TF-IDF and User Ratings Analysis
"""

import re
import logging
import inspect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from IPython.display import display

# Configuring the logging
logging.basicConfig(filename = "log_movie_recommendations.log",
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)


# https://files.grouplens.org/datasets/movielens/ml-25m.zip

def handle_errors(func):
    """
    A decorator for enhancing error handling within functions.

    Args:
        func (function): The function to which error handling capabilities are added.

    Returns:
        function: A wrapped version of the original function with improved error handling.

    Description:
        The `handle_errors` decorator is designed to make functions more resilient to exceptions by
        adding error-catching capabilities. When applied to a function, it encapsulates that 
        function within a try-except block, allowing for graceful error handling.

    Functionality:
        1. The decorator takes one argument, `func`, which should be the function to enhance with
        error handling.
        2. A `wrapper` function is defined within the decorator to wrap the original 
        function (`func`).
        3. When the decorated function is called, the `wrapper` function intercepts the call.
        4. Inside the try block, the `wrapper` function attempts to execute the original 
        function with any provided arguments and keyword arguments (`*args` and `**kwargs`).
        5. If an exception of type `Exception` (or its subclasses) occurs during the execution of
        the original function, it is caught, and the caught exception is stored in the 
        variable `error`.
        6. In case of an exception, the decorator logs an error message indicating that an error
        occurred in the specific function (`func.__name__`) at a particular line number and includes 
        information about the error.
        7. After handling the error (or if no error occurred), the decorator returns 
        the result of the original function's execution. If an error was caught,
        `result` is set to `None`.

    Usage:
        To use the `handle_errors` decorator, apply it to any function where you want to add
        error-handling functionality. For example:
        
        @handle_errors
        def my_function():
            # Your function code here
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except NameError as error:
            caller_info = inspect.getframeinfo(inspect.currentframe().f_back)
            logging.error("An error occurred in func._name__:%s at line %s "
                        "and the following error: %s.\nPlease if check your imported dataset",
                        {func.__name__}, {caller_info.lineno}, {error})
            result = None
        except TypeError as error:
            caller_info = inspect.getframeinfo(inspect.currentframe().f_back)
            logging.error("An error occurred in func._name__:%s at line %s "
                        "and the following error: %s.\nPay attention to type of your argument.",
                        {func.__name__}, {caller_info.lineno}, {error})
            result = None
        except FileNotFoundError as error:
            caller_info = inspect.getframeinfo(inspect.currentframe().f_back)
            logging.error("An error occurred in func._name__:%s at line %s "
                        "and the following error: %s.\nMust check the folder",
                        {func.__name__}, {caller_info.lineno}, {error})
            result = None
        except AttributeError as error:
            caller_info = inspect.getframeinfo(inspect.currentframe().f_back)
            logging.error("An error occurred in func._name__:%s at line %s "
                        "and the following error: %s.\nMust check wether the object has attribute",
                        {func.__name__}, {caller_info.lineno}, {error})
            result = None
        return result
    return wrapper


def clean_title(title):
    """
    Clean up titles by removing special characters
    
    Args:
        tittle (str): The movie title to clean up.

    Returns:
        title (str): The movie title without any special characters

    Description:
        1. Select the complement of the set of common characters, i.e. special characters
        2. Just remove the special without swap for another characters.
    """

    try:
        title = re.sub("[^a-zA-Z0-9 ]", "", title)
        return title
    except TypeError as error:
        caller_info = inspect.getframeinfo(inspect.currentframe().f_back)
        logging.error("An error occurred in func._name__:%s at line %s "
                    "and the following error: %s.\nYour argument is invalid, "
                    "check if it's a string.",
                    {clean_title.__name__}, {caller_info.lineno}, {error})
        # logging.error("Your argument is invalid, check if it's a string, %s", error)

        logging.debug("Leaving from clean tittle function returning the None.")

        return None

try:
    # Using Pandas to open a csv file
    movies = pd.read_csv("movies.csv")
    logging.info("Your successfully import the data!")
    # Take a glimpse of the data
    movies.head()


    # Apply the cleaning on the dataset and save in another column
    movies["clean_title"] = movies["title"].apply(clean_title)
    print(movies)

except FileNotFoundError:
    # print("Can't found the file, please check the path.")
    logging.error("Can't found the file, please check the path.")


@handle_errors
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
    logging.debug("Entered in search function.")

    # Create a TF-IDF Vectorizer with unigram and bigram support
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    logging.info("Successfuly nstatiate a TfidVectorizer with a unigram and bigram")

    # Transform the clean movie titles into a TF-IDF matrix
    tfidf = vectorizer.fit_transform(movies["clean_title"])
    logging.info("Successfully used in your titles")

    title = clean_title(title)
    logging.info("Successfully got clean titles.")

    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    logging.debug("Exiting the search function returning similar tittles.")

    return results


@handle_errors
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
    logging.debug("Entered in search function.")

    ratings = pd.read_csv("../../../ratings.csv")
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

    logging.debug("Exiting the search function returning similar movies.")

    return rec_percentages.head(10).merge(movies, left_index=True,
                                        right_on="movieId")[["score", "title", "genres"]]


# print(find_similar_movies(89745))

@handle_errors
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
    title = data
    if len(title) > 5:
        results = search(title)
        movie_id = results.iloc[0]["movieId"]
        display(find_similar_movies(movie_id))


movie_tittle = input("Digit a title of a movie: ")
on_type(movie_tittle)
