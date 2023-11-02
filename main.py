# Code comment below (# %%) wraps file in an Jypter code cell and runs in interactive window. Use for visualization of dataframes
# %%
from sportsipy.nfl.boxscore import Boxscores  # Retrieve a dictionary which contains a major game data of all games being played on a particular day.
from sportsipy.nfl.boxscore import Boxscore  # Detailed information about the final statistics for a game.
import pandas as pd
import sys

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

    # List of week range. Note that lastweek is not inclusive so we add 1
    weeks_list = list(range(firstweek, lastweek + 1))

    # Instantiate schedule dataframe
    schedule_SUM_DF = pd.DataFrame()

    # For each week of the season
    # for w in range(len(weeks_list)):
    for w in weeks_list:

        # Create key in the string format "Week-Year"
        date_str = str(w) + '-' + str(year)

        # Create Boxscores Object for current week w     
        week_w_BOX = Boxscores(w, year)

        # Instantiate dataframe for current week w
        week_games_SUM_DF = pd.DataFrame()

        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):
            # Create dataframe out of select game statistic keys
            game_SUM_DF = pd.DataFrame(week_w_BOX.games[date_str][g], index=[0], columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr'])

            # Add week # to each index
            game_SUM_DF['week'] = w

            # Concat current game to list of this weeks game
            week_games_SUM_DF = pd.concat([week_games_SUM_DF, game_SUM_DF])

        # Concat current game to season long dataframe
        schedule_SUM_DF = pd.concat([schedule_SUM_DF, week_games_SUM_DF]).reset_index().drop(columns='index')

    return schedule_SUM_DF

def game_data(game_df, game_stats):
    try:
        away_team_df = game_df[['away_name', 'away_abbr', 'away_score']].rename(columns = {'away_name': 'team_name', 'away_abbr': 'team_abbr', 'away_score': 'score'})
        home_team_df = game_df[['home_name','home_abbr', 'home_score']].rename(columns = {'home_name': 'team_name', 'home_abbr': 'team_abbr', 'home_score': 'score'})
        try:
            if game_df.loc[0,'away_score'] > game_df.loc[0,'home_score']:
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [1], 'game_lost' : [0]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [1]}),left_index = True, right_index = True)
            elif game_df.loc[0,'away_score'] < game_df.loc[0,'home_score']:
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [1]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [1], 'game_lost' : [0]}),left_index = True, right_index = True)
            else: 
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [0]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [0], 'game_lost' : [0]}),left_index = True, right_index = True)
        except TypeError:    
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won' : [np.nan], 'game_lost' : [np.nan]}),left_index = True, right_index = True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won' : [np.nan], 'game_lost' : [np.nan]}),left_index = True, right_index = True)     

        away_stats_df = game_stats.dataframe[['away_first_downs', 'away_fourth_down_attempts',
            'away_fourth_down_conversions', 'away_fumbles', 'away_fumbles_lost',
            'away_interceptions', 'away_net_pass_yards', 'away_pass_attempts',
            'away_pass_completions', 'away_pass_touchdowns', 'away_pass_yards',
            'away_penalties', 'away_points', 'away_rush_attempts',
            'away_rush_touchdowns', 'away_rush_yards', 'away_third_down_attempts',
            'away_third_down_conversions', 'away_time_of_possession',
            'away_times_sacked', 'away_total_yards', 'away_turnovers',
            'away_yards_from_penalties', 'away_yards_lost_from_sacks']].reset_index().drop(columns ='index').rename(columns = {
            'away_first_downs': 'first_downs', 'away_fourth_down_attempts':'fourth_down_attempts',
            'away_fourth_down_conversions':'fourth_down_conversions' , 'away_fumbles': 'fumbles', 'away_fumbles_lost': 'fumbles_lost',
            'away_interceptions': 'interceptions', 'away_net_pass_yards':'net_pass_yards' , 'away_pass_attempts': 'pass_attempts',
            'away_pass_completions':'pass_completions' , 'away_pass_touchdowns': 'pass_touchdowns', 'away_pass_yards': 'pass_yards',
            'away_penalties': 'penalties', 'away_points': 'points', 'away_rush_attempts': 'rush_attempts',
            'away_rush_touchdowns': 'rush_touchdowns', 'away_rush_yards': 'rush_yards', 'away_third_down_attempts': 'third_down_attempts',
            'away_third_down_conversions': 'third_down_conversions', 'away_time_of_possession': 'time_of_possession',
            'away_times_sacked': 'times_sacked', 'away_total_yards': 'total_yards', 'away_turnovers': 'turnovers',
            'away_yards_from_penalties':'yards_from_penalties', 'away_yards_lost_from_sacks': 'yards_lost_from_sacks'})
        home_stats_df = game_stats.dataframe[['home_first_downs', 'home_fourth_down_attempts',
            'home_fourth_down_conversions', 'home_fumbles', 'home_fumbles_lost',
            'home_interceptions', 'home_net_pass_yards', 'home_pass_attempts',
            'home_pass_completions', 'home_pass_touchdowns', 'home_pass_yards',
            'home_penalties', 'home_points', 'home_rush_attempts',
            'home_rush_touchdowns', 'home_rush_yards', 'home_third_down_attempts',
            'home_third_down_conversions', 'home_time_of_possession',
            'home_times_sacked', 'home_total_yards', 'home_turnovers',
            'home_yards_from_penalties', 'home_yards_lost_from_sacks']].reset_index().drop(columns = 'index').rename(columns = {
            'home_first_downs': 'first_downs', 'home_fourth_down_attempts':'fourth_down_attempts',
            'home_fourth_down_conversions':'fourth_down_conversions' , 'home_fumbles': 'fumbles', 'home_fumbles_lost': 'fumbles_lost',
            'home_interceptions': 'interceptions', 'home_net_pass_yards':'net_pass_yards' , 'home_pass_attempts': 'pass_attempts',
            'home_pass_completions':'pass_completions' , 'home_pass_touchdowns': 'pass_touchdowns', 'home_pass_yards': 'pass_yards',
            'home_penalties': 'penalties', 'home_points': 'points', 'home_rush_attempts': 'rush_attempts',
            'home_rush_touchdowns': 'rush_touchdowns', 'home_rush_yards': 'rush_yards', 'home_third_down_attempts': 'third_down_attempts',
            'home_third_down_conversions': 'third_down_conversions', 'home_time_of_possession': 'time_of_possession',
            'home_times_sacked': 'times_sacked', 'home_total_yards': 'total_yards', 'home_turnovers': 'turnovers',
            'home_yards_from_penalties':'yards_from_penalties', 'home_yards_lost_from_sacks': 'yards_lost_from_sacks'})
                
        away_team_df = pd.merge(away_team_df, away_stats_df,left_index = True, right_index = True)
        home_team_df = pd.merge(home_team_df, home_stats_df,left_index = True, right_index = True)
        try:
            away_team_df['time_of_possession'] = (int(away_team_df['time_of_possession'].loc[0][0:2]) * 60) + int(away_team_df['time_of_possession'].loc[0][3:5])
            home_team_df['time_of_possession'] = (int(home_team_df['time_of_possession'].loc[0][0:2]) * 60) + int(home_team_df['time_of_possession'].loc[0][3:5])
        except TypeError:
            away_team_df['time_of_possession'] = np.nan
            home_team_df['time_of_possession'] = np.nan
    except TypeError:
        away_team_df = pd.DataFrame()
        home_team_df = pd.DataFrame()
    
    return away_team_df, home_team_df

def main():
    # Tests for sportsipy_submodule_summary function
    if(False):
        sportsipy_submodule_summary()

    # Tests for get_schedule function. Week 13 of 2022 has a Tie so its a good query.
    if(False):
        display(get_schedule(2022, 13, 13))

    # Tests for game_data function
    if(True):
        # Create Boxscores Object for the 2022 season weeks 13 - 13
        weekXthruY_BOX = Boxscores(13, 2022, 13)

        # Get week 13, game 6's URI
        weekX_gameY_URI = weekXthruY_BOX.games['13-2022'][6]['boxscore']

        # Create Detailed Boxscore object using URI
        weekX_gameY_BOX = Boxscore(weekX_gameY_URI)

        # Print brief summary of a single game within Boxscore's scope.
        # This is the short version of a Boxscore Object
        weekX_gameY_SUM = weekXthruY_BOX.games['13-2022'][6]
        weekX_gameY_SUM_DF = pd.DataFrame.from_dict([weekX_gameY_SUM])

        away_STATS_DF, home_STATS_DF = clean_game_data(weekX_gameY_SUM_DF, weekX_gameY_BOX)
        display(away_STATS_DF)
        display(home_STATS_DF)

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Exit: Keyboard Interrupted")
        sys.exit(0)

# %%
