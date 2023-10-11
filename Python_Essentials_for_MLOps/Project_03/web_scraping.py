"""
NBA Data Scraping and Processing Module

This module provides functions for scraping NBA-related data, including MVP awards,
player statistics, and team standings, from the Basketball-Reference website. It also
includes functions for processing the downloaded data and saving it to CSV files.

Functions:
- download_page(url, filename): Download a web page and save it to a file.
- download_page_with_selenium(url, filename): Download a web page using Selenium and save 
it to a file.
- process_and_save_data(years, data_type, data_id, csv_filename): Process HTML data, extract 
relevant information, and save it to a CSV file.

Usage:
1. Use the download functions to retrieve data from specified URLs.
2. Use the process_and_save_data function to process and save data in CSV files.

Note:
- Ensure that you have the required libraries installed, such as requests, BeautifulSoup, 
pandas, and Selenium.
- Configure Chrome and ChromeDriver properly for Selenium-based scraping.
- Follow the naming convention for HTML data files: {data_type}/{year}.html
- Adjust the URLs and file paths as needed for your specific use case.
"""

import logging
import inspect
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver

# Setup the URLs
all_years = list(range(1991, 2022))
URL_START = "https://www.basketball-reference.com/awards/awards_{}.html"
PLAYER_STATS_URL = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"
TEAM_STATS_URL = "https://www.basketball-reference.com/leagues/NBA_{}_standings.html"

# Configuring the logging
logging.basicConfig(filename = "web_scraping.log",
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)

DELAY = 20

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

@handle_errors
def download_page(url, filename):
    """
    Download the content from a given URL and save it to a file.

    Args:
        url (str): The URL to download content from.
        filename (str): The name of the file to save the content to.

    This function sends an HTTP GET request to the specified URL, waits for DELAY seconds 
    (for demonstration purposes),
    and saves the received data as text to the specified file. It's essential to
    provide a valid URL and a filename
    to save the downloaded content.
    """
    data = requests.get(url, timeout=10)
    time.sleep(DELAY)
    with open(filename, "w+", encoding='utf-8') as file:
        file.write(data.text)

@handle_errors
def download_page_with_selenium(url, filename):
    """
    Download the content of a web page using Selenium and save it to a file.

    Args:
        url (str): The URL of the web page to download.
        filename (str): The name of the file to save the web page content to.

    This function uses Selenium to open a web page, scroll to the end of the page,
    wait for DELAY seconds (for demonstration purposes), and then saves the page's HTML source 
    code to the specified file. It's important to have Selenium and
    ChromeDriver properly configured for this function to work.

    Note:
        Ensure that you have the Chrome browser installed and the appropriate
        ChromeDriver executable for your version of Chrome. You also need to install 
        the Selenium library.
    """
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=webdriver.ChromeService(), options=option)
    driver.get(url)
    driver.execute_script("window.scrollTo(1,10000)")
    time.sleep(DELAY)
    with open(filename, "w+", encoding='utf-8') as file:
        file.write(driver.page_source)
    driver.quit()

@handle_errors
def process_and_save_data(years, data_type, data_id, csv_filename):
    """
    Process HTML data, extract relevant information, and save it to a CSV file.

    Args:
        years (list): A list of years for which data is being processed.
        data_type (str): The type of data being processed (e.g., 'mvp', 'player', 'team').
        data_id (str): The HTML element ID containing the data of interest.
        csv_filename (str): The name of the CSV file to save the processed data to.

    This function processes HTML data from multiple years, extracts the relevant information 
    based on the provided
    data_type and data_id, and combines the data into a single DataFrame. The processed data 
    is then saved to a CSV file with the specified filename.

    Args:
        years (list): A list of years for which data is being processed.
        data_type (str): The type of data being processed (e.g., 'mvp', 'player', 'team').
        data_id (str): The HTML element ID containing the data of interest.
        csv_filename (str): The name of the CSV file to save the processed data to.

    Note:
        - Ensure that the HTML data files exist in the specified directory and follow 
        the naming convention:
          {data_type}/{year}.html
        - The BeautifulSoup library and pandas library are required for 
        processing and saving the data.
    """
    dfs = []
    for year in years:
        with open(f"{data_type}/{year}.html", encoding='utf-8') as file:
            page = file.read()

        soup = BeautifulSoup(page, 'html.parser')
        soup.find('tr', class_="over_header").decompose()

        logging.debug("A problem in %s in archive of %s.", data_type, year)

        data_table = soup.find_all(id=data_id)[0]
        data_df = pd.read_html(str(data_table))[0]
        data_df["Year"] = year
        dfs.append(data_df)

    combined_data = pd.concat(dfs)
    combined_data.to_csv(csv_filename)

# Download all pages of MVP
logging.info("Start to download all pages of mvps")
for one_year in all_years:
    full_url = URL_START.format(one_year)
    download_page(full_url, f"mvp/{one_year}.html")
logging.info("Finish downloads of mvps")

# Download all pages of players using selenium
logging.info("Start to download all pages of players")
for one_year in all_years:
    full_url = PLAYER_STATS_URL.format(one_year)
    download_page_with_selenium(full_url, f"player/{one_year}.html")
logging.info("Finish downloads of players")

# Download all pages of teams
logging.info("Start to download all pages of teams")
for one_year in all_years:
    full_url = TEAM_STATS_URL.format(one_year)
    download_page(full_url, f"team/{one_year}.html")
logging.info("Finish downloads of teams")

# Process and save in csv files
process_and_save_data(all_years, "mvp", "mvp", "mvps.csv")
process_and_save_data(all_years, "player", "per_game_stats", "players.csv")
process_and_save_data(all_years, "team", "divs_standings_E", "teams.csv")
