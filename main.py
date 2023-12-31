# Code comment below (# %%) wraps file in an Jypter code cell and runs in interactive window. Use for visualization of dataframes
from sportsipy.nfl.boxscore import Boxscores  # Retrieve a dictionary which contains a major game data of all games being played on a particular day.
from sportsipy.nfl.boxscore import Boxscore  # Detailed information about the final statistics for a game.
import pandas as pd
import numpy as np
import sys
import pickle
from IPython.display import display, HTML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, brier_score_loss
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statistics 

OFFLINE_MODE = True
PREPROCESSING = False

"""
Coding Standard
    Suffix Summary:
        _BOX                    Either a Boxscores Object (set of games) or a Boxscore Object (single game)
                                EX. Boxscores(W, YYYY, W) or Boxscore(URI)
        
        _SUM                    Brief summary of a set of games or a single game
                                EX. Boxscores.games or Boxscores.games[W-YYYY][Game #]

        _STATS                  A collection of stats that isnt directly a _BOX or a _SUM.
        
        _DF                     Any object/variable coverted to a pandas.DataFrame           
"""

def store_data(year, firstweek, lastweek):
    """Pickle's a Boxscore object for each week in the query. Stores a CSV of a Boxscore.dataframe for each game within the query.

    Args:
        year (int): Year of schedule query
        firstweek (int): Starting week of schedule query (inclusive)
        lastweek (int): Ending week of schedule query (inclusive)
    """
    weeks_list = list(range(firstweek, lastweek + 1))

    for w in weeks_list:
        # Create key in the string format "W-YYYY"
        date_str = str(w) + '-' + str(year)

        # Create and store Boxscores Object for current week w
        week_w_BOX = Boxscores(w, year)
        print(week_w_BOX)
        print(w)
        with open(f'.\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'wb') as file: 
            pickle.dump(week_w_BOX, file) 
        
        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):

            # Extract game URI, create Boxscore object, store its dataframe
            game_URI = week_w_BOX.games[date_str][g]['boxscore']
            game_BOX_DF = Boxscore(game_URI).dataframe
            print(g)
            print(game_URI)
            game_BOX_DF.to_csv(f'.\\Data\\GameBoxscore\\Boxscore_{game_URI}.csv')
        
def sportsipy_submodule_summary():
    """
    Instantiate and print some of the various Data Types within the sportsipy submodule.
    This function is only used for debugging and helping developers understand each module.

    Args:
        None

    Returns:
        None
    """

    # Create Boxscores Object for the 2022 season weeks x - y. Week 13 of 2022 has a Tie so its a good query.
    weekXthruY_BOX = Boxscores(13, 2022, 13)
    print("Boxscores Class: ")
    print(weekXthruY_BOX)

    # Print dictionary of all games played within the Boxscores' scope (Weeks 1 - 8)
    # Format {Week: [{Array of dictionaries that contain game info}]}
    weekXthruY_SUM = weekXthruY_BOX.games
    print("\nBoxscores.games: ")
    print(weekXthruY_SUM)

    # Print brief summary of a single game within Boxscore's scope.
    # This is the short version of a Boxscore Object
    weekX_gameY_SUM = weekXthruY_BOX.games['13-2022'][6]
    weekX_gameY_SUM_DF = pd.DataFrame.from_dict([weekX_gameY_SUM])
    print("\nGame Summary: ")
    print(weekX_gameY_SUM_DF.to_string())

    # Get week 1, game 1's URI
    weekX_gameY_URI = weekXthruY_BOX.games['13-2022'][6]['boxscore']
    print("\nGame 1 URI: ")
    print(weekX_gameY_URI)

    # Create Detailed Boxscore object using URI
    weekX_gameY_BOX = Boxscore(weekX_gameY_URI)
    print("\nBoxscore: ")
    print(weekX_gameY_BOX)

    # Create dataframe out of week X, game Y's boxscore
    weekX_gameY_BOX_DF = weekX_gameY_BOX.dataframe
    print("\nBoxscore Week X Game Y DataFrame: ")
    print(weekX_gameY_BOX_DF.to_string())

def get_schedule(year, firstweek, lastweek):
    """
    Create a pandas.DataFrame of games played within a specified timeframe.

    Args:
        year (int): Year of schedule query
        firstweek (int): Starting week of schedule query (inclusive)
        lastweek (int): Ending week of schedule query (inclusive)

    Returns:
        schedule_SUM_DF (pandas.DataFrame): A DataFrame of all games played within the scope of the query, 
                            where each row corresponds to a single game.
    """

    # NFL schedule expanded in 2021. Handle bad input of 18th week prior to 2021 season
    if (year < 2021 & lastweek > 17):
        lastweek = 17

    # List of week range. Note that lastweek is not inclusive so we add 1
    weeks_list = list(range(firstweek, lastweek + 1))

    # Instantiate schedule dataframe
    schedule_SUM_DF = pd.DataFrame()

    # For each week of the season
    # for w in range(len(weeks_list)):
    for w in weeks_list:

        # Create key in the string format "W-YYYY"
        date_str = str(w) + '-' + str(year)

        if(OFFLINE_MODE):
            # Load and deserialize Boxscores(W, YYYY)
            with open(f'.\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'rb') as file: 
                week_w_BOX = pickle.load(file) 
        else:
            # Create Boxscores Object for current week w     
            week_w_BOX = Boxscores(w, year)

        # Instantiate dataframe for current week w
        week_games_SUM_DF = pd.DataFrame()

        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):
            # Create dataframe out of select game statistic keys
            game_SUM_DF = pd.DataFrame(week_w_BOX.games[date_str][g], index=[0], columns=['away_name', 'away_abbr','away_score', 'home_name', 'home_abbr', 'home_score', 'winning_name', 'winning_abbr'])
            
            # Add week # to each index
            game_SUM_DF['week'] = w
            
            # Concat current game to list of this weeks game
            week_games_SUM_DF = pd.concat([week_games_SUM_DF, game_SUM_DF])

        # Concat current game to season long dataframe
        schedule_SUM_DF = pd.concat([schedule_SUM_DF, week_games_SUM_DF]).reset_index().drop(columns='index')

    return schedule_SUM_DF

def get_clean_game_data(game_SUM_DF, game_BOX_DF):
    """
        Clean data from a single game and return two pandas.DataFrames that has the cleaned data in a more usable form.
        'away_' and 'home_' prefixes are removed, time of possession is converted into seconds, game result is converted from a string of abbr to a 1 or 0. 

    Args:
        game_SUM_DF (pandas.Dataframe): Single game summary DataFrame
        game_BOX_DF (pandas.Dataframe): A Boxscore data frame

    Returns:
        away_STATS_DF, home_STATS_DF (pandas.DataFrame, pandas.DataFrame): A cleaned DataFrame of all stats for each team provided in the two arguments. 
                                                                             Each Return corresponds to that teams stats in that singular game.
    """
    try:
        # Create away DataFrame out of only away team stats and remove 'away_' prefix from columns
        away_SUM_DF = game_SUM_DF.filter(regex="^away_")
        away_SUM_DF.columns = away_SUM_DF.columns.str.removeprefix("away_")

        # Create home DataFrame out of only home team stats and remove 'home_' prefix from columns
        home_SUM_DF = game_SUM_DF.filter(regex="^home_")
        home_SUM_DF.columns = home_SUM_DF.columns.str.removeprefix("home_")

        # If away team won, set won/lost fields
        if game_SUM_DF.loc[0,'away_score'] > game_SUM_DF.loc[0,'home_score']:
            away_SUM_DF = pd.merge(away_SUM_DF, pd.DataFrame({'game_won' : [1], 'game_lost' : [0]}),left_index = True, right_index = True)
            home_SUM_DF = pd.merge(home_SUM_DF, pd.DataFrame({'game_won' : [0], 'game_lost' : [1]}),left_index = True, right_index = True)
        
        # If home team won, set won/lost fields
        elif game_SUM_DF.loc[0,'away_score'] < game_SUM_DF.loc[0,'home_score']:
            away_SUM_DF = pd.merge(away_SUM_DF, pd.DataFrame({'game_won' : [0], 'game_lost' : [1]}),left_index = True, right_index = True)
            home_SUM_DF = pd.merge(home_SUM_DF, pd.DataFrame({'game_won' : [1], 'game_lost' : [0]}),left_index = True, right_index = True)
        
        # If tie, set won/lost fields
        else: 
            away_SUM_DF = pd.merge(away_SUM_DF, pd.DataFrame({'game_won' : [0], 'game_lost' : [0]}),left_index = True, right_index = True)
            home_SUM_DF = pd.merge(home_SUM_DF, pd.DataFrame({'game_won' : [0], 'game_lost' : [0]}),left_index = True, right_index = True)  
       
        # Create away Boxscore DF out of only away team stats, reset index, and remove 'away_' prefix from columns
        away_BOX_DF = game_BOX_DF.filter(regex="^away_")
        away_BOX_DF = away_BOX_DF.reset_index().drop(columns = 'index')
        away_BOX_DF.columns = away_BOX_DF.columns.str.removeprefix("away_")

        # Create home Boxscore DF out of only home team stats, reset index, and remove 'home_' prefix from columns
        home_BOX_DF = game_BOX_DF.filter(regex="^home_")
        home_BOX_DF = home_BOX_DF.reset_index().drop(columns = 'index')
        home_BOX_DF.columns = home_BOX_DF.columns.str.removeprefix("home_")

        # Combine summary DataFrame and Boxscore DataFrame         
        away_STATS_DF = pd.merge(away_SUM_DF, away_BOX_DF, left_index = True, right_index = True)
        home_STATS_DF = pd.merge(home_SUM_DF, home_BOX_DF, left_index = True, right_index = True)

        # Convert time of posession from MM:SS to seconds
        away_STATS_DF['time_of_possession'] = (int(away_STATS_DF['time_of_possession'].loc[0][0:2]) * 60) + int(away_STATS_DF['time_of_possession'].loc[0][3:5])
        home_STATS_DF['time_of_possession'] = (int(home_STATS_DF['time_of_possession'].loc[0][0:2]) * 60) + int(home_STATS_DF['time_of_possession'].loc[0][3:5])

    # Handle various errors        
    except (TypeError, KeyError, ValueError, AttributeError) as err:
        print(err)
        away_STATS_DF = pd.DataFrame()
        home_STATS_DF = pd.DataFrame()
    
    return away_STATS_DF, home_STATS_DF

def get_game_data_for_weeks(weeks_list, year):
    """
        Get a DataFrame of cleaned data of a single teams's stats in a single game. The DataFrame will contain info for all teams that played a game in the scope of the arguments. 
        This method repeatedly calls and aggregates data from get_clean_game_data() which returns clean game data for a single game. 

    Args:
        weeks_list (List of ints): Weeks of clean game data query
        year (int): Year of clean game data query

    Returns:
        weeks_games_STATS_DF (pandas.DataFrame): A DataFrame that contains cleaned data for all games played within the scope of the query, 
                            where each row corresponds to a single teams stats in that game.
    """
    
    weeks_games_STATS_DF = pd.DataFrame()
    
    for w in weeks_list:
        
        # Create key in the string format "W-YYYY"
        date_str = str(w) + '-' + str(year)

        if(OFFLINE_MODE):
            # Load and deserialize Boxscores(W, YYYY)
            with open(f'.\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'rb') as file: 
                week_w_BOX = pickle.load(file) 
        else:
            # Create Boxscores Object for current week w     
            week_w_BOX = Boxscores(w, year)

        # Instantiate dataframe for current week w
        week_games_SUM_DF = pd.DataFrame()
        
        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):

            # Extract game URI and create Boxscore object
            game_URI = week_w_BOX.games[date_str][g]['boxscore']
            if(OFFLINE_MODE):
                game_BOX_DF = pd.read_csv(f'.\\Data\\GameBoxscore\\Boxscore_{game_URI}.csv', index_col=0)
            else:
                game_BOX_DF = Boxscore(game_URI).dataframe

            # Create datafame out of select game statistics
            game_SUM_DF = pd.DataFrame(week_w_BOX.games[date_str][g], index = [0])

            # Clean data for each game summary and boxscore
            away_team_df, home_team_df = get_clean_game_data(game_SUM_DF, game_BOX_DF)

            # Add week # to each index
            away_team_df['week'] = w
            home_team_df['week'] = w
    
            # Concat current game to list of this week's games
            week_games_SUM_DF = pd.concat([week_games_SUM_DF, away_team_df])
            week_games_SUM_DF = pd.concat([week_games_SUM_DF, home_team_df])
        
        # Concat this week's games to overall dataframe
        weeks_games_STATS_DF = pd.concat([weeks_games_STATS_DF, week_games_SUM_DF])
                        
    return weeks_games_STATS_DF
    

def agg_weekly_data(schedule_DF, weeks_games_SUM_DF, current_week, weeks_list):
    """Creates table of each game of all weeks in the query where the stats associated with the given game is the difference between the two opponents' weeks_list averages up to the week the game was played. 
        E.x., Week 10 of 2023 Buffalo hosted the Denver. If your weeks list is [1, 2, ... , 10] then their entry into the dataframe would be (Denver week 1-9 average statistics) - (Buffalo week 1-9 average statistics).

    Args:
        schedule_DF (pandas.Dataframe): Output of get_schedule.
        weeks_games_SUM_DF (pandas.Dataframe): Output of get_game_data_for_weeks. Function call to get_game_data_for_weeks must use same weeks_list argument as this function. 
        current_week (int): Defines schedule query, must be greater than highest value in weeks_list. In future can be used to compare two teams on any given week. 
        weeks_list (List of ints): weeks of query

    Returns:
        pandas.Dataframe: Difference between two opponents' averages up to the week the game was played.  
    """
    # Extract schedule for only weeks that have been played
    schedule_DF = schedule_DF[schedule_DF.week <= current_week]

    # Instantiate statistic aggrigation dataframe
    agg_weekly_diff_DF = pd.DataFrame()

    # For each week that has been played
    for w in weeks_list:

        # Extract current week's matchups
        single_week_games_DF = schedule_DF[schedule_DF.week == w]

        # Create dataframe of stats we want to average for all weeks up to week w
        teams_weekly_avg_DF = weeks_games_SUM_DF[weeks_games_SUM_DF.week < w].drop(columns = ['score', 'week', 'game_won', 'game_lost'])
  
        # Group each team's week info into one line (pandas.groupby)
        # Compute running average of per-game stats (pandas.mean)
        teams_weekly_avg_DF = teams_weekly_avg_DF.groupby(by=["name", "abbr"]).mean().reset_index().round(3)

        # Create dataframe of stats we want to average for all weeks up to week w
        win_loss_df = weeks_games_SUM_DF[weeks_games_SUM_DF.week < w][["name", "abbr",'game_won', 'game_lost']]

        # Calculate win percentage
        win_loss_df = win_loss_df.groupby(by=["name", "abbr"]).sum().reset_index()
        win_loss_df['win_perc'] = (win_loss_df['game_won'] / (win_loss_df['game_won'] + win_loss_df['game_lost'])).round(3)
        win_loss_df = win_loss_df.drop(columns = ['game_won', 'game_lost'])

        # Handle Week 1 Ties
        win_loss_df['win_perc'] = win_loss_df['win_perc'].fillna(0)

        # Calculate 3rd and 4th down conversion rates
        try:
            teams_weekly_avg_DF['fourth_down_perc'] = teams_weekly_avg_DF['fourth_down_conversions'] / teams_weekly_avg_DF['fourth_down_attempts']
        except ZeroDivisionError:
            teams_weekly_avg_DF['fourth_down_perc'] = 0
            teams_weekly_avg_DF['fourth_down_perc'] = teams_weekly_avg_DF['fourth_down_perc'].fillna(0)
        
        try:
            teams_weekly_avg_DF['third_down_perc'] = teams_weekly_avg_DF['third_down_conversions'] / teams_weekly_avg_DF['third_down_attempts']
        except ZeroDivisionError:
            teams_weekly_avg_DF['third_down_perc'] = 0

        # Remove counting stat represenations of 3rd and 4th down statistics
        teams_weekly_avg_DF['fourth_down_perc'] = teams_weekly_avg_DF['fourth_down_perc'].fillna(0)
        teams_weekly_avg_DF['third_down_perc'] = teams_weekly_avg_DF['third_down_perc'].fillna(0)
        teams_weekly_avg_DF = teams_weekly_avg_DF.drop(columns = ['fourth_down_attempts', 'fourth_down_conversions', 'third_down_attempts', 'third_down_conversions'])

        # Add win percentage DF to aggregation DF
        teams_weekly_avg_DF = pd.merge(win_loss_df,teams_weekly_avg_DF, on = ['name', 'abbr'])

        # Merge intersection of season long averages into upcoming weeks schedule
        away_df = pd.merge(single_week_games_DF,teams_weekly_avg_DF.add_prefix('away_'), on = ['away_name', 'away_abbr'])
        home_df = pd.merge(single_week_games_DF,teams_weekly_avg_DF.add_prefix('home_'), on = ['home_name', 'home_abbr'])    
        teams_weekly_avg_DF = pd.merge(away_df,home_df, on = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr', 'week', 'home_score', 'away_score'])         
        
        # Create differential of of Away-Home season long averages
        teams_weekly_avg_diff_DF = teams_weekly_avg_DF
        teams_weekly_avg_diff_DF['win_perc_dif'] = teams_weekly_avg_DF['away_win_perc'] - teams_weekly_avg_DF['home_win_perc']         
        teams_weekly_avg_diff_DF['first_downs_dif'] = teams_weekly_avg_DF['away_first_downs'] - teams_weekly_avg_DF['home_first_downs']         
        teams_weekly_avg_diff_DF['fumbles_dif'] = teams_weekly_avg_DF['away_fumbles'] - teams_weekly_avg_DF['home_fumbles']         
        teams_weekly_avg_diff_DF['interceptions_dif'] = teams_weekly_avg_DF['away_interceptions'] - teams_weekly_avg_DF['home_interceptions']         
        teams_weekly_avg_diff_DF['net_pass_yards_dif'] = teams_weekly_avg_DF['away_net_pass_yards'] - teams_weekly_avg_DF['home_net_pass_yards']         
        teams_weekly_avg_diff_DF['pass_attempts_dif'] = teams_weekly_avg_DF['away_pass_attempts'] - teams_weekly_avg_DF['home_pass_attempts']         
        teams_weekly_avg_diff_DF['pass_completions_dif'] = teams_weekly_avg_DF['away_pass_completions'] - teams_weekly_avg_DF['home_pass_completions']         
        teams_weekly_avg_diff_DF['pass_touchdowns_dif'] = teams_weekly_avg_DF['away_pass_touchdowns'] - teams_weekly_avg_DF['home_pass_touchdowns']         
        teams_weekly_avg_diff_DF['pass_yards_dif'] = teams_weekly_avg_DF['away_pass_yards'] - teams_weekly_avg_DF['home_pass_yards']         
        teams_weekly_avg_diff_DF['penalties_dif'] = teams_weekly_avg_DF['away_penalties'] - teams_weekly_avg_DF['home_penalties']        
        teams_weekly_avg_diff_DF['points_dif'] = teams_weekly_avg_DF['away_points'] - teams_weekly_avg_DF['home_points']         
        teams_weekly_avg_diff_DF['rush_attempts_dif'] = teams_weekly_avg_DF['away_rush_attempts'] - teams_weekly_avg_DF['home_rush_attempts']
        teams_weekly_avg_diff_DF['rush_touchdowns_dif'] = teams_weekly_avg_DF['away_rush_touchdowns'] - teams_weekly_avg_DF['home_rush_touchdowns']
        teams_weekly_avg_diff_DF['rush_yards_dif'] = teams_weekly_avg_DF['away_rush_yards'] - teams_weekly_avg_DF['home_rush_yards']
        teams_weekly_avg_diff_DF['time_of_possession_dif'] = teams_weekly_avg_DF['away_time_of_possession'] - teams_weekly_avg_DF['home_time_of_possession']
        teams_weekly_avg_diff_DF['times_sacked_dif'] = teams_weekly_avg_DF['away_times_sacked'] - teams_weekly_avg_DF['home_times_sacked']
        teams_weekly_avg_diff_DF['total_yards_dif'] = teams_weekly_avg_DF['away_total_yards'] - teams_weekly_avg_DF['home_total_yards']
        teams_weekly_avg_diff_DF['turnovers_dif'] = teams_weekly_avg_DF['away_turnovers'] - teams_weekly_avg_DF['home_turnovers'] 
        teams_weekly_avg_diff_DF['yards_from_penalties_dif'] = teams_weekly_avg_DF['away_yards_from_penalties'] - teams_weekly_avg_DF['home_yards_from_penalties']
        teams_weekly_avg_diff_DF['yards_lost_from_sacks_dif'] = teams_weekly_avg_DF['away_yards_lost_from_sacks'] - teams_weekly_avg_DF['home_yards_lost_from_sacks']
        teams_weekly_avg_diff_DF['fourth_down_perc_dif'] = teams_weekly_avg_DF['away_fourth_down_perc'] - teams_weekly_avg_DF['home_fourth_down_perc']
        teams_weekly_avg_diff_DF['third_down_perc_dif'] = teams_weekly_avg_DF['away_third_down_perc'] - teams_weekly_avg_DF['home_third_down_perc']
        teams_weekly_avg_diff_DF = teams_weekly_avg_diff_DF.drop(
            columns=['away_win_perc', 'away_first_downs', 'away_fumbles', 'away_fumbles_lost', 'away_interceptions', 'away_net_pass_yards', 'away_pass_attempts', 'away_pass_completions', 'away_pass_touchdowns', 'away_pass_yards',
                     'away_penalties', 'away_points', 'away_rush_attempts', 'away_rush_touchdowns', 'away_rush_yards', 'away_time_of_possession', 'away_times_sacked', 'away_total_yards', 'away_turnovers', 'away_yards_from_penalties',
                     'away_yards_lost_from_sacks', 'away_fourth_down_perc', 'away_third_down_perc', 'home_win_perc', 'home_first_downs', 'home_fumbles', 'home_fumbles_lost', 'home_interceptions', 'home_net_pass_yards', 'home_pass_attempts',
                     'home_pass_completions', 'home_pass_touchdowns', 'home_pass_yards', 'home_penalties', 'home_points', 'home_rush_attempts', 'home_rush_touchdowns', 'home_rush_yards', 'home_time_of_possession', 'home_times_sacked',
                     'home_total_yards', 'home_turnovers', 'home_yards_from_penalties', 'home_yards_lost_from_sacks', 'home_fourth_down_perc', 'home_third_down_perc'])

        if(PREPROCESSING):
            if (teams_weekly_avg_diff_DF['winning_name'].isnull().values.any()):
                teams_weekly_avg_diff_DF = teams_weekly_avg_diff_DF.dropna(subset="winning_abbr")

            teams_weekly_avg_diff_DF['result'] = teams_weekly_avg_diff_DF['winning_name'] == teams_weekly_avg_diff_DF['away_name']
            teams_weekly_avg_diff_DF['result'] = teams_weekly_avg_diff_DF['result'].astype('float')
            teams_weekly_avg_diff_DF = teams_weekly_avg_diff_DF.drop(columns = ['winning_name', 'winning_abbr']) 
            agg_weekly_diff_DF = pd.concat([agg_weekly_diff_DF, teams_weekly_avg_diff_DF])
            agg_weekly_diff_DF = agg_weekly_diff_DF.reset_index().drop(columns = 'index')

        else:
            # If game has been played add win/loss flag to the dataframe
            if (teams_weekly_avg_diff_DF['winning_name'].isnull().values.any()):
                teams_weekly_avg_diff_DF['result'] = np.nan
                print(f"Week {w} games have not finished yet.")        
            else:
                teams_weekly_avg_diff_DF['result'] = teams_weekly_avg_diff_DF['winning_name'] == teams_weekly_avg_diff_DF['away_name']
                teams_weekly_avg_diff_DF['result'] = teams_weekly_avg_diff_DF['result'].astype('float')
                teams_weekly_avg_diff_DF = teams_weekly_avg_diff_DF.drop(columns = ['winning_name', 'winning_abbr']) 
                agg_weekly_diff_DF = pd.concat([agg_weekly_diff_DF, teams_weekly_avg_diff_DF])
                agg_weekly_diff_DF = agg_weekly_diff_DF.reset_index().drop(columns = 'index')

    return agg_weekly_diff_DF

def get_elo(year):
    """Returns filtered week by week 583 Elo rankings for given year

    Args:
        year (int): year of query

    Returns:
        pandas.Dataframe: Filtered week by week 583 Elo rankings for given year
    """
    # Get stored CSV and filter for 2022 season regular season
    elo_DF = pd.read_csv(f'.\\Data\\nfl_elo.csv')
    elo_DF = elo_DF[elo_DF['playoff'].isna()]
    elo_DF = elo_DF[elo_DF['season'] == year]
    
    # Drop unwanted columns
    elo_DF = elo_DF.drop(columns = ['season', 'neutral' ,'playoff', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post',
           'qbelo1_pre', 'qbelo2_pre', 'qb1', 'qb2', 'qb1_adj', 'qb2_adj', 'qbelo_prob1', 'qbelo_prob2',
           'qb1_game_value', 'qb2_game_value', 'qb1_value_post', 'qb2_value_post',
           'qbelo1_post', 'qbelo2_post', 'score1', 'score2'])
    
    # Rename team abbreviations to match pro-football-reference abbreviations
    elo_DF['team1'] = elo_DF['team1'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
           'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
           'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
           'SEA', 'OAK'],
            ['kan','jax','car', 'rav', 'buf', 'min', 'det', 'atl', 'nwe', 'was', 
            'cin', 'nor', 'sfo', 'ram', 'nyg', 'den', 'cle', 'clt', 'oti', 'nyj', 
             'tam','mia', 'pit', 'phi', 'gnb', 'chi', 'dal', 'crd', 'sdg', 'htx', 'sea', 'rai' ])
    elo_DF['team2'] = elo_DF['team2'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
           'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
           'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
           'SEA', 'OAK'],
            ['kan','jax','car', 'rav', 'buf', 'min', 'det', 'atl', 'nwe', 'was', 
            'cin', 'nor', 'sfo', 'ram', 'nyg', 'den', 'cle', 'clt', 'oti', 'nyj', 
             'tam','mia', 'pit', 'phi', 'gnb', 'chi', 'dal', 'crd', 'sdg', 'htx', 'sea', 'rai' ])

    return elo_DF


def get_spread(year, firstweek, lastweek):
    """
    Create a pandas.DataFrame of Las Vegas spreads for every game played within a specified timeframe.

    Args:
        year (int): Year of schedule query
        firstweek (int): Starting week of schedule query (inclusive)
        lastweek (int): Ending week of schedule query (inclusive)

    Returns:
        schedule_SUM_DF (pandas.DataFrame): A DataFrame of all games played within the scope of the query, 
                            where each row corresponds to a single game and they include the Las Vegas spread.

    """

    # NFL schedule expanded in 2021. Handle bad input of 18th week prior to 2021 season
    if (year < 2021 and lastweek > 17):
        lastweek = 17


    # List of week range. Note that lastweek is not inclusive so we add 1
    weeks_list = list(range(firstweek, lastweek + 1))

    # Instantiate schedule dataframe
    schedule_SUM_DF = pd.DataFrame()
    spread_DF = pd.DataFrame()

    # For each week of the season
    # for w in range(len(weeks_list)):
    for w in weeks_list:

        # Create key in the string format "W-YYYY"
        date_str = str(w) + '-' + str(year)

        if(OFFLINE_MODE):
            # Load and deserialize Boxscores(W, YYYY)
            with open(f'.\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'rb') as file: 
                week_w_BOX = pickle.load(file) 
        else:
            # Create Boxscores Object for current week w     
            week_w_BOX = Boxscores(w, year)

        # Instantiate dataframe for current week w
        week_games_SUM_DF = pd.DataFrame()

        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):

            # Extract game URI and create Boxscore object
            game_URI = week_w_BOX.games[date_str][g]['boxscore']
            if(OFFLINE_MODE):
                game_BOX_DF = pd.read_csv(f'.\\Data\\GameBoxscore\\Boxscore_{game_URI}.csv', index_col=0)
            else:
                game_BOX_DF = Boxscore(game_URI).dataframe

            # Create dataframe out of select game statistic keys
            game_SUM_DF = pd.DataFrame(week_w_BOX.games[date_str][g], index=[0], columns=['away_name', 'away_abbr','away_score', 'home_name', 'home_abbr', 'home_score', 'winning_name', 'winning_abbr'])
            
            # Create dataframe out of select game statistic keys
            spread_DF = game_BOX_DF.filter(['vegas_line'])

            # Handle pickem games
            if spread_DF['vegas_line'].loc[spread_DF.index[0]].strip() == "Pick":
                spread_double = 0

            else:   
                # If not pickem extract score and adjust sign       
                spread_DF[['favorite', 'spread']] = spread_DF['vegas_line'].str.split('-', n=1, expand=True)
            
                spread_double = float(spread_DF['spread'].loc[spread_DF.index[0]])
               
                if (spread_DF['favorite'].loc[spread_DF.index[0]].strip() == game_SUM_DF['home_name'].loc[game_SUM_DF.index[0]].strip()):
                    spread_double = spread_double * -1
            
            game_SUM_DF['vegas_line'] = spread_double
            
            # Add week # to each index
            game_SUM_DF['week'] = w

            # Concat current game to list of this weeks game
            week_games_SUM_DF = pd.concat([week_games_SUM_DF, game_SUM_DF])

        # Concat current game to season long dataframe
        schedule_SUM_DF = pd.concat([schedule_SUM_DF, week_games_SUM_DF]).reset_index().drop(columns='index')

    return schedule_SUM_DF

def merge_rankings(weekly_agg_DF,elo_DF, spread_DF):
    """Merge agg_weekly_data output with get_elo output. Calculates difference between opponent's elo and adds it to agg_weekly_data's output which 
    contains a statistical difference between two opponents' averages up to the week the game was played.

    Args:
        weekly_agg_DF (pandas.Dataframe): agg_weekly_data output. Statistical difference between two opponent's averages up to the week the game was played.
        elo_DF (pandas.Dataframe): get_elo output. Filtered week by week 583 Elo rankings for given year.
        spread_DF (pandas.Dataframe): get_spread output. Filtered week by week Las Vegas gambling spreads.

    Returns:
        pandas.Dataframe: Week by week statistical difference between two opponents including elo rating
    """
    # Merge tables based on intersection of abbreviations
    
    weekly_agg_DF = pd.merge(weekly_agg_DF, elo_DF, how = 'inner', left_on = ['home_abbr', 'away_abbr'], right_on = ['team1', 'team2']).drop(columns = ['date', 'team1', 'team2'])


    if(PREPROCESSING):
        weekly_agg_DF = pd.merge(weekly_agg_DF, spread_DF, how = 'inner', on = ['home_abbr', 'away_abbr', 'week', 'away_name', 'home_name', 'home_score', 'away_score']).drop(columns = ['winning_name', 'winning_abbr'])

    # Calculate difference between opponent's elo
    weekly_agg_DF['elo_dif'] = weekly_agg_DF['elo2_pre'] - weekly_agg_DF['elo1_pre']
    weekly_agg_DF['qb_dif'] = weekly_agg_DF['qb2_value_pre'] - weekly_agg_DF['qb1_value_pre']
    #weekly_agg_DF['score_dif'] = weekly_agg_DF['away_score'] - weekly_agg_DF['home_score']

    # Drop unused elo stats
    weekly_agg_DF = weekly_agg_DF.drop(columns = ['elo1_pre', 'elo2_pre', 'qb1_value_pre', 'qb2_value_pre'])
    
    return weekly_agg_DF

def prep_model_data(current_week, weeks_list, year):
    """ Returns a training set of games that have happened and a test set of games that will happen. 
    Both sets contain week by week statistical and elo differences between opponents. 

    Args:
        current_week (int): Week of test data (games that will happen)
        weeks_list (List of ints): Weeks of training game data (games that have happened)
        year (int): Year of data query

    Returns:
        test_DF (pandas.Dataframe), training_DF (pandas.Dataframe): Week by week statistical and elo differences between two opponents. 
                                                                    training_DF is the set of games that have happened
                                                                    test_DF is the set of games that will happen
    """
    # Get schedule and weekly summary of games
    current_week = current_week + 1
    schedule_DF  = get_schedule(year, 1, 18)

    # Returns DataFrame that has single game counting stats. Each row is one team's game result
    weeks_games_SUM_DF = get_game_data_for_weeks(weeks_list, year)

    # Week by week statistical difference between two opponents
    weekly_agg_DF = agg_weekly_data(schedule_DF, weeks_games_SUM_DF, current_week, weeks_list)

    # Get week by week team elo ratings and week by week spread
    elo_DF = get_elo(year)
    spread_DF = get_spread(year, 1, 18)

    # Merge elo ratings with week by week statistical difference dataframe (weekly_agg_DF)
    weekly_agg_DF = merge_rankings(weekly_agg_DF, elo_DF, spread_DF)
    
    # Seperate training data (games that have happened) from test data (games that will happen)
    current_week = current_week - 1
    training_DF = weekly_agg_DF[weekly_agg_DF['week'] < current_week]
    test_DF = weekly_agg_DF[weekly_agg_DF.week == current_week]

    return test_DF, training_DF


def displayFunc(y_pred_data_list, test_data_DF, conv_scores):
    """Displays weekly projections in a more human redable format. 

    Args:
        y_pred_data_list (array): Array of predicted win probability
        test_data_DF (pandas.Dataframe): test data set
        conv_scores (array): Array of predicted win probability as a point differental
    """
    print(print(f"{'Away Team' : <20}{'Win Probability Over' : ^30}{'Home Team' : <25}{"Vegas Spread" : ^7}"))
    for g in range(len(y_pred_data_list)):
        win_prob = round(y_pred_data_list[g], 2)
        spread = float(-1*conv_scores[g])
        away_team = test_data_DF.reset_index().drop(columns='index').loc[g, 'away_name']
        home_team = test_data_DF.reset_index().drop(columns='index').loc[g, 'home_name']
        print(f"{away_team : <20}{win_prob : ^30}{home_team : <25} ({'{:+}'.format(spread) : ^7})")


def correlationDimensionalityReduction(x_training_data_DF, x_test_data_DF):
    """Removes highly correlated features from training and test data

    Args:
        x_training_data_DF (pandas.Dataframe): Training data set 
        x_test_data_DF (pandas.Dataframe): Test data set

    Returns:
        x_training_data_cleaned_DF (pandas.Dataframe), x_test_data_cleaned_DF (pandas.Dataframe): Training and test data sets with highly correlated features removed
    """
    # Create correlation out of training data
    corr = x_training_data_DF.corr()
            
    # Mask top half of correlation heatmap so redundant values are not printed
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr = corr.mask(mask)

    # Size and create heatmap 
    fig, ax = plt.subplots(figsize=(15,15))     
    cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
    sns.heatmap(corr, center=0, cmap=cmap, linewidths=1, annot=True, fmt=".2f", ax=ax)

    # Extract features that are over 90% correlated to another feature
    to_drop = [c for c in corr.columns if any(corr[c].abs() > 0.90)]
    
    # Drop features that are over 90% correlated to another feature
    print(to_drop)
    return (x_training_data_DF.drop(columns = to_drop), x_test_data_DF.drop(columns = to_drop))

def generalDimensionalityReduction(x_training_data_DF, x_test_data_DF):
    """Removes select features from training and test data

    Args:
        x_training_data_DF (pandas.Dataframe): Training data set 
        x_test_data_DF (pandas.Dataframe): Test data set

    Returns:
        x_training_data_cleaned_DF (Array), x_test_data_cleaned_DF (pandas.Dataframe): Training and test data sets with select features removed
    """
   
    to_drop = ['quality', 'importance', 'total_rating', 'win_perc_dif']
    return (x_training_data_DF.drop(columns = to_drop), x_test_data_DF.drop(columns = to_drop))


def displayWinPerc(completed_games_DF):
    """Visualize histogram of win_perc_diff stat of the training data

    Args:
        completed_games_DF (pandas.Dataframe): Training dataset that will be analyzed
    """
    # Format diagram
    plt.figure(figsize = (10,10))
    plt.title("Histogram of win percentage difference in each game")
    
    # Calculate mean and std of win percentage
    mean, std = norm.fit(completed_games_DF['win_perc_dif']) 
    x_min = completed_games_DF['win_perc_dif'].min()
    x_max = completed_games_DF['win_perc_dif'].max()

    # Create and plot probability density function 
    sns.histplot(x = completed_games_DF['win_perc_dif'], edgecolor = None, color = 'lightblue', bins = 15, stat = 'density', label = '2022 Win Diff Probablity Density Function')
    
    # Create and plot normal distribution function
    x_range = np.arange(x_min, x_max, 0.001)
    y_range = norm.pdf(x_range, mean, std)
    plt.plot(x_range, y_range, label = "Normal Probability Density Function")
    
    # Show diagram
    plt.legend(loc='upper right')        
    plt.show()

def getScores(y_pred_data_list, test_data_DF):
    actual_scores = list(range(len(y_pred_data_list)))
    for g in range(len(y_pred_data_list)):
        actual_scores[g] = test_data_DF.reset_index().drop(columns='index').loc[g, 'away_score'] - test_data_DF.reset_index().drop(columns='index').loc[g, 'home_score']
    return actual_scores

def conversion(pred_list):
    # Define a dictionary to map ranges to values
    range_favored = {
        (0.51, 0.5249): -1,
        (0.525, 0.5349): -1.5,
        (0.535, 0.5449): -2,
        (0.545, 0.5939): -2.5,
        (0.594, 0.6429): -3,
        (0.643, 0.6579): -3.5,
        (0.658, 0.6729): -4,
        (0.673, 0.6809): -4.5,
        (0.681, 0.6899): -5,
        (0.69, 0.7069): -5.5,
        (0.7070, 0.7239): -6,
        (0.724, 0.7519): -6.5,
        (0.752, 0.7809): -7,
        (0.781, 0.7909): -7.5,
        (0.791, 0.8019): -8,
        (0.8020, 0.8069): -8.5,
        (0.8070, 0.8109): -9,
        (0.8110, 0.8359): -9.5,
        (0.8360, 0.8599): -10,
        (0.86, 0.8709): -10.5,
        (0.871, 0.8819): -11,
        (0.882, 0.8849): -11.5,
        (0.885, 0.8869): -12,
        (0.887, 0.8929): -12.5,
        (0.893, 0.8999): -13,
        (0.9, 0.9239): -13.5,
        (0.924, 0.9489): -14,
        (0.949, 0.9559): -14.5,
        (0.956, 0.9629): -15,
        (0.963, 0.9809): -15.5,
        (0.981, 0.9999): -16,
        (1.0, 1.0): -16.5
        
    }
    range_underdog = {
        (0.488, 0.4751): 1,
        (0.475, 0.4651): 1.5,
        (0.465, 0.4551): 2,
        (0.455, 0.4061): 2.5,
        (0.406, 0.3571): 3,
        (0.357, 0.3421): 3.5,
        (0.342, 0.3271): 4,
        (0.327, 0.3191): 4.5,
        (0.319, 0.3111): 5,
        (0.311, 0.2941): 5.5,
        (0.294, 0.2771): 6,
        (0.277, 0.2481): 6.5,
        (0.248, 0.2191): 7,
        (0.219, 0.2091): 7.5,
        (0.209, 0.1981): 8,
        (0.198, 0.1931): 8.5,
        (0.193, 0.1891): 9,
        (0.189, 0.1641): 9.5,
        (0.164, 0.1401): 10,
        (0.14, 0.1291): 10.5,
        (0.129, 0.1181): 11,
        (0.118, 0.1161): 11.5,
        (0.116, 0.1131): 12,
        (0.113, 0.1071): 12.5,
        (0.107, 0.1001): 13,
        (0.1, 0.0761): 13.5,
        (0.076, 0.0511): 14,
        (0.051, 0.0441): 14.5,
        (0.044, 0.0371): 15,
        (0.037, 0.0191): 15.5,
        (0.019, 0.0001): 16,
        (0.0, 0.0): 16.5
    }
    conv_list = []

    for pred in pred_list:
        rounded_pred = round(pred, 2)
        mapped_value = None

        for range_key, value in range_favored.items():
            if range_key[0] <= rounded_pred <= range_key[1]:
                mapped_value = value
                break

        for range_key, value in range_underdog.items():
            if range_key[0] >= rounded_pred >= range_key[1]:
                mapped_value = value
                break

        # If the prediction is exactly 0.5, set mapped_value to 0
        if 0.488 < rounded_pred < 0.51:
            mapped_value = 0

        # Append the mapped value if found, else the original prediction
        conv_list.append(mapped_value if mapped_value is not None else pred)

    return conv_list

def main():
    if (True):
        firstweek = 1
        current_week = 18
        weeks_list = list(range(firstweek, current_week + 1))
        year = 2021
        future_games__2021_DF, completed_games_2021_DF = prep_model_data(current_week, weeks_list, year)

        firstweek = 1
        current_week = 18
        weeks_list = list(range(firstweek, current_week + 1))
        year = 2022
        future_games__2022_DF, completed_games_2022_DF = prep_model_data(current_week, weeks_list, year)

        # Concat 2020 and 2021 training data
        train_data_DF = pd.concat([completed_games_2021_DF, future_games__2021_DF]).reset_index().drop(columns='index')
        test_data_DF = pd.concat([completed_games_2022_DF, future_games__2022_DF]).reset_index().drop(columns='index')

        # Separate input training data from result training outcome
        x_training_data_DF = train_data_DF.drop(columns=['away_name', 'away_abbr','away_score','home_name', 'home_abbr', 'home_score', 'week', 'result'])
        y_training_data_DF = train_data_DF[['result']]
        x_test_data_DF = test_data_DF.drop(columns=['away_name', 'away_abbr', 'away_score', 'home_name', 'home_abbr', 'home_score', 'week', 'result'])
        y_test_data_DF = test_data_DF[['result']]


        if(PREPROCESSING):
            x_training_data_DF, x_test_data_DF = correlationDimensionalityReduction(x_training_data_DF, x_test_data_DF)
            x_training_data_DF, x_test_data_DF = generalDimensionalityReduction(x_training_data_DF, x_test_data_DF)
            displayWinPerc(train_data_DF)

        # Create linear regression function
        clf = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                                 intercept_scaling=1, class_weight='balanced', random_state=None,
                                 solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)
        
        
        # Fit model according to training data        
        clf.fit(x_training_data_DF, np.ravel(y_training_data_DF.values))
        y_pred_data_list = clf.predict_proba(x_test_data_DF)
        y_pred_data_list = y_pred_data_list[:, 1]

        actual_scores = getScores(y_pred_data_list, test_data_DF)
        conv_scores = conversion(y_pred_data_list)
        displayFunc(y_pred_data_list, test_data_DF, conv_scores)

        # ROC Analysis
        fpr, tpr, thresholds = roc_curve(y_test_data_DF, y_pred_data_list)
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # Plot Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test_data_DF, y_pred_data_list)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend(loc="lower right")
        plt.show()

        # Plot Precision/Recall vs Threshold Curve
        plt.figure()
        plt.plot(thresholds, precision[1:], color='darkorange', lw=2, label='Precision')
        plt.plot(thresholds, recall[1:], color='navy', lw=2, label='Recall')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.title('Precision Recall Curve')
        plt.legend(loc="lower right")
        plt.show()

        y_true = y_test_data_DF['result'].astype(int).values
        y_prob = y_pred_data_list

        # Calculate Brier Score
        brier_score = brier_score_loss(y_true, y_prob)
        print(f"Brier Score: {round(brier_score, 3)}")

        mse = np.square(np.subtract(actual_scores, conv_scores)).mean()
        rmse = math.sqrt(mse)
        print(f"Root Mean Square Error: {round(rmse, 3)}")
    
        # Check our predictions against the completed test data games.
        # Round predicted probablity of a victory to a 1 or 0
        straight_up_perc = accuracy_score(y_test_data_DF, np.round(y_pred_data_list))
        print(f"Straight Up Accuracy:  {'{:.2%}'.format(straight_up_perc)}")

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Exit: Keyboard Interrupted")
        sys.exit(0)

