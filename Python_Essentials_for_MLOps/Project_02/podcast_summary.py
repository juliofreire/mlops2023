"""D"""

import os
import json
import inspect
import logging
import requests
import xmltodict
import pendulum
from airflow.decorators import dag, task
from airflow.providers.sqlite.operators.sqlite import SQLExecuteQueryOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# Configuring the logging
logging.basicConfig(filename = "podcast_summary.log",
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)

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
        return result
    return wrapper



PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000

@dag(
    dag_id='podcast_summary',
    schedule="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)

@handle_errors
def podcast_summary():
    """
    Automated Podcast Summary Pipeline

    This function defines an automated pipeline for processing podcast episodes.
    The pipeline consists of several tasks, each performing a specific operation:

    1. Create Database Table:
        - Creates a SQLite database table named 'episodes' if it doesn't already exist.
        - The table structure includes columns for episode metadata and transcript storage.

    2. Get Episodes:
        - Retrieves podcast episodes from a specified URL using an HTTP GET request.
        - Parses the XML feed to extract episode information and returns a list of episodes.

    3. Load Episodes:
        - Compares the retrieved episodes with those already stored in the database.
        - Identifies new episodes and inserts them into the 'episodes' table.

    4. Download Episodes:
        - Downloads audio files (MP3) associated with the podcast episodes.
        - Ensures that each audio file is downloaded only once.

    5. Speech-to-Text Transcription:
        - Transcribes the audio content of downloaded episodes into text using Vosk ASR.
        - Inserts the transcripts into the 'transcript' column of the 'episodes' table.

    Parameters:
        - None

    Usage:
        Call the 'podcast_summary' function to initiate the automated pipeline.

    Note:
        - This pipeline depends on various libraries and configurations such as requests,
        xmltodict, SqliteHook, Vosk ASR model, and more.
        - Ensure that all dependencies are installed and configured before running the pipeline.

"""
    create_database = SQLExecuteQueryOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    @handle_errors
    @task()
    def get_episodes():
        data = requests.get(PODCAST_URL, timeout=10)
        feed = xmltodict.parse(data.text)
        episodes = feed["rss"]["channel"]["item"]
        print(f"Found {len(episodes)} episodes.")
        return episodes

    podcast_episodes = get_episodes()
    create_database.set_downstream(podcast_episodes)

    @task()
    def load_episodes(episodes):
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []
        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append([episode["link"], episode["title"],
                                    episode["pubDate"], episode["description"], filename])

        hook.insert_rows(table='episodes', rows=new_episodes,
                        target_fields=["link", "title", "published", "description", "filename"])

        return new_episodes

    # new_episodes = load_episodes(podcast_episodes)

    @handle_errors
    @task()
    def download_episodes(episodes):
        audio_files = []
        for episode in episodes:
            name_end = episode["link"].split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(EPISODE_FOLDER, filename)
            if not os.path.exists(audio_path):
                print(f"Downloading {filename}")
                audio = requests.get(episode["enclosure"]["@url"], timeout=10)
                with open(audio_path, "wb+") as file:
                    file.write(audio.content)
            audio_files.append({
                "link": episode["link"],
                "filename": filename
            })
        return audio_files

    # audio_files = download_episodes(podcast_episodes)

    @handle_errors
    @task()
    def speech_to_text():
        hook = SqliteHook(sqlite_conn_id="podcasts")
        untranscribed_episodes = hook.get_pandas_df(
            "SELECT * from episodes WHERE transcript IS NULL;")

        model = Model(model_name="vosk-model-en-us-0.22-lgraph")
        rec = KaldiRecognizer(model, FRAME_RATE)
        rec.SetWords(True)

        for _, row in untranscribed_episodes.iterrows():
            print(f"Transcribing {row['filename']}")
            filepath = os.path.join(EPISODE_FOLDER, row["filename"])
            mp3 = AudioSegment.from_mp3(filepath)
            mp3 = mp3.set_channels(1)
            mp3 = mp3.set_frame_rate(FRAME_RATE)

            step = 20000
            transcript = ""
            for i in range(0, len(mp3), step):
                print(f"Progress: {i/len(mp3)}")
                segment = mp3[i:i+step]
                rec.AcceptWaveform(segment.raw_data)
                result = rec.Result()
                text = json.loads(result)["text"]
                transcript += text
            hook.insert_rows(table='episodes', rows=[[row["link"], transcript]],
                            target_fields=["link", "transcript"], replace=True)

    #Uncomment this to try speech to text (may not work)
    #speech_to_text(audio_files, new_episodes)

SUMMARY = podcast_summary()
