"""
NBA MVP Data Analysis Module

This module contains functions for loading, processing, merging, and analyzing NBA MVP data.
It offers capabilities to handle missing data, process player information, and extract insights
related to the highest-scoring players and other statistical trends over the years.

Functions:
    - load_data: Load data from multiple CSV files and return DataFrames along with a dictionary of team nicknames.
    - process_players: Process player data by handling cases with multiple team entries.
    - merge_data: Merge and prepare the combined data for analysis.
    - analyze_data: Analyze and extract insights from the merged data.

Example:
    mvps, players, teams, nicknames = load_data()
    players = process_players(players)
    trains = merge_data(mvps, players, teams)
    highest_scorings, highest_scoring_by_years, player_count_by_years = analyze_data(trains)

Additional analysis or visualization can be performed after running these functions.

"""
import pandas as pd

def load_data():
    """
    Load data from multiple CSV files and return DataFrames along with a 
    dictionary of team nicknames.

    Returns:
        pd.DataFrame: DataFrame containing MVP data (Player, Year, Pts Won, Pts Max, Share).
        pd.DataFrame: DataFrame containing player data (Player, Year, and other statistics).
        pd.DataFrame: DataFrame containing team data (Team, Year, and various statistics).
        dict: A dictionary mapping team abbreviations to full team names.
    """
    # Load MVP data
    mvp = pd.read_csv("mvps.csv")
    mvp = mvp[["Player", "Year", "Pts Won", "Pts Max", "Share"]]

    # Load player data
    player = pd.read_csv("players.csv")
    del player["Unnamed: 0"]
    del player["Rk"]
    player["Player"] = player["Player"].str.replace("*", "", regex=False)

    # Load team data
    team = pd.read_csv("teams.csv")
    team = team[~team["W"].str.contains("Division")].copy()
    team["Team"] = team["Team"].str.replace("*", "", regex=False)

    # Load team nickname mappings
    nickname = {}
    with open("nicknames.txt", encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines[1:]:
            abbrev, name = line.replace("\n", "").split(",")
            nickname[abbrev] = name

    return mvp, player, team, nickname

def process_players(player):
    """
    Process player data by handling cases with multiple team entries.

    Args:
        player (pd.DataFrame): DataFrame containing player data, including team information.

    Returns:
        pd.DataFrame: Processed player data with single team entries for each player in each year.

    This function takes a DataFrame containing player data and addresses cases where a 
    player has multiple team entries for a single year. It returns a processed DataFrame 
    where each player's entry for a year represents their primary team, or "TOT" if they 
    played for multiple teams in that year.

    Args:
        player (pd.DataFrame): DataFrame containing player data, including team information.
    Returns:
        pd.DataFrame: Processed player data with single team entries for each player in each year.
    """
    def single_team(data):
        if data.shape[0] == 1:
            return data
        else:
            row = data[data["Tm"] == "TOT"]
            row["Tm"] = data.iloc[-1, :]["Tm"]
            return row

    player = player.groupby(["Player", "Year"]).apply(single_team)
    player.index = player.index.droplevel()
    player.index = player.index.droplevel()
    return player

def merge_data(mvp, player, team):
    """
    Merge and prepare the combined data for analysis.

    Args:
        mvp (pd.DataFrame): DataFrame containing MVP award data.
        player (pd.DataFrame): DataFrame containing player data.
        team (pd.DataFrame): DataFrame containing team data.

    Returns:
        pd.DataFrame: Merged and processed data ready for analysis.

    This function merges MVP, player, and team data, filling missing MVP data with zeros. 
    It then performs data cleaning and conversion, including handling the "GB" column. 
    The resulting DataFrame is prepared for further analysis.

    Args:
        mvp (pd.DataFrame): DataFrame containing MVP award data.
        player (pd.DataFrame): DataFrame containing player data.
        team (pd.DataFrame): DataFrame containing team data.
    Returns:
        pd.DataFrame: Merged and processed data ready for analysis.
    """
    combined = player.merge(mvp, how="outer", on=["Player", "Year"])
    combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)

    train = combined.merge(team, how="outer", on=["Team", "Year"])
    del train["Unnamed: 0"]
    train = train.apply(pd.to_numeric, errors='ignore')
    train["GB"] = pd.to_numeric(train["GB"].str.replace("—", "0"))

    return train

def analyze_data(train):
    """
    Analyze and extract insights from the merged data.

    Args:
        train (pd.DataFrame): Merged and processed data.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series: Dataframes for highest-scoring players, 
        highest scorers by year, and a series for player count by year.

    This function takes the merged and processed data and performs analysis to extract insights.
    It identifies the highest-scoring players with at least 70 games, the highest scorer by year,
    and counts the number of players by year.

    Args:
        train (pd.DataFrame): Merged and processed data.
    Returns:
        pd.DataFrame: Highest-scoring players with at least 70 games.
        pd.DataFrame: Highest scorer by year.
        pd.Series: Player count by year.
    """
    highest_scoring = train[train["G"] > 70].sort_values("PTS", ascending=False).head(10)
    highest_scoring_by_year = train.groupby("Year").apply(lambda x: x.sort_values("PTS",
                                                    ascending=False).head(1))
    player_count_by_year = train.groupby("Year").apply(lambda x: x.shape[0])

    return highest_scoring, highest_scoring_by_year, player_count_by_year

mvps, players, teams, nicknames = load_data()
players = process_players(players)
trains = merge_data(mvps, players, teams)
highest_scorings, highest_scoring_by_years, player_count_by_years = analyze_data(trains)
