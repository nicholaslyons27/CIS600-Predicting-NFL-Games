# Code comment below (# %%) wraps file in an Jypter code cell and runs in interactive window. Use for visualization of dataframes
# %%
from sportsipy.nfl.boxscore import Boxscores  # Retrieve a dictionary which contains a major game data of all games being played on a particular day.
from sportsipy.nfl.boxscore import Boxscore  # Detailed information about the final statistics for a game.
import pandas as pd
import sys
import pickle

OFFLINE_MODE = True

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
        with open(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'wb') as file: 
            pickle.dump(week_w_BOX, file) 
        
        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):

            # Extract game URI, create Boxscore object, store its dataframe
            game_URI = week_w_BOX.games[date_str][g]['boxscore']
            game_BOX_DF = Boxscore(game_URI).dataframe
            print(g)
            print(game_URI)
            game_BOX_DF.to_csv(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\GameBoxscore\\Boxscore_{game_URI}.csv')
    
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

        # Create key in the string format "W-YYYY"
        date_str = str(w) + '-' + str(year)

        if(OFFLINE_MODE):
            # Load and deserialize Boxscores(W, YYYY)
            with open(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'rb') as file: 
                week_w_BOX = pickle.load(file) 
        else:
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

def get_clean_game_data(game_SUM_DF, game_BOX_DF):
    """
        Clean data from a single game and return two pandas.DataFrames that has the cleaned data in a more usable form.
        'away_' and 'home_' prefixes are removed, time of possession is converted into seconds, game result is converted from a string of team_abbr to a 1 or 0. 

    Args:
        game_SUM_DF (_type_): Single game summary DataFrame
        game_BOX_DF (_type_): A Boxscore data frame

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

def get_game_data_for_weeks(weeks, year):
    """
        Get a DataFrame of cleaned data of a single teams's stats in a single game. The DataFrame will contain info for all teams that played a game in the scope of the arguments. 
        This method repeatedly calls and aggregates data from get_clean_game_data() which returns clean game data for a single game. 

    Args:
        weeks (List of Ints): Weeks of clean game data query
        year (_type_): Year of clean game data query

    Returns:
        weeks_games_STATS_DF (pandas.DataFrame): A DataFrame that contains cleaned data for all games played within the scope of the query, 
                            where each row corresponds to a single teams stats in that game.
    """
    
    weeks_games_STATS_DF = pd.DataFrame()
    
    for w in weeks:
        
        # Create key in the string format "W-YYYY"
        date_str = str(w) + '-' + str(year)

        if(OFFLINE_MODE):
            # Load and deserialize Boxscores(W, YYYY)
            with open(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'rb') as file: 
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
                game_BOX_DF = pd.read_csv(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\GameBoxscore\\Boxscore_{game_URI}.csv')
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
    

def agg_weekly_data(schedule_df,weeks_games_df,current_week,weeks):
    schedule_df = schedule_df[schedule_df.week < current_week]
    agg_games_df = pd.DataFrame()
    for w in range(1,len(weeks)):
        games_df = schedule_df[schedule_df.week == weeks[w]]
        agg_weekly_df = weeks_games_df[weeks_games_df.week < weeks[w]].drop(columns = ['score','week','game_won', 'game_lost']).groupby(by=["team_name", "team_abbr"]).mean().reset_index()
        win_loss_df = weeks_games_df[weeks_games_df.week < weeks[w]][["team_name", "team_abbr",'game_won', 'game_lost']].groupby(by=["team_name", "team_abbr"]).sum().reset_index()
        win_loss_df['win_perc'] = win_loss_df['game_won'] / (win_loss_df['game_won'] + win_loss_df['game_lost'])
        win_loss_df = win_loss_df.drop(columns = ['game_won', 'game_lost'])
        
        try:
            agg_weekly_df['fourth_down_perc'] = agg_weekly_df['fourth_down_conversions'] / agg_weekly_df['fourth_down_attempts']
        except ZeroDivisionError:
            agg_weekly_df['fourth_down_perc'] = 0
            agg_weekly_df['fourth_down_perc'] = agg_weekly_df['fourth_down_perc'].fillna(0)
        
        try:
            agg_weekly_df['third_down_perc'] = agg_weekly_df['third_down_conversions'] / agg_weekly_df['third_down_attempts']
        except ZeroDivisionError:
            agg_weekly_df['third_down_perc'] = 0
        
        agg_weekly_df['third_down_perc'] = agg_weekly_df['third_down_perc'].fillna(0)
        agg_weekly_df = agg_weekly_df.drop(columns = ['fourth_down_attempts', 'fourth_down_conversions', 'third_down_attempts', 'third_down_conversions'])
        agg_weekly_df = pd.merge(win_loss_df,agg_weekly_df,left_on = ['team_name', 'team_abbr'], right_on = ['team_name', 'team_abbr'])
        away_df = pd.merge(games_df,agg_weekly_df,how = 'inner', left_on = ['away_name', 'away_abbr'], right_on = ['team_name', 'team_abbr']).drop(columns = ['team_name', 'team_abbr']).rename(columns = {'win_perc': 'away_win_perc','first_downs': 'away_first_downs', 'fumbles': 'away_fumbles', 'fumbles_lost':'away_fumbles_lost', 'interceptions':'away_interceptions','net_pass_yards': 'away_net_pass_yards', 'pass_attempts':'away_pass_attempts', 'pass_completions':'away_pass_completions', 'pass_touchdowns':'away_pass_touchdowns', 'pass_yards':'away_pass_yards', 'penalties':'away_penalties', 'points':'away_points', 'rush_attempts':'away_rush_attempts', 'rush_touchdowns':'away_rush_touchdowns', 'rush_yards':'away_rush_yards', 'time_of_possession':'away_time_of_possession', 'times_sacked':'away_times_sacked', 'total_yards':'away_total_yards', 'turnovers':'away_turnovers', 'yards_from_penalties':'away_yards_from_penalties', 'yards_lost_from_sacks': 'away_yards_lost_from_sacks', 'fourth_down_perc': 'away_fourth_down_perc', 'third_down_perc':'away_third_down_perc'})         
        home_df = pd.merge(games_df,agg_weekly_df,how = 'inner', left_on = ['home_name', 'home_abbr'], right_on = ['team_name', 'team_abbr']).drop(columns = ['team_name', 'team_abbr']).rename(columns = {'win_perc': 'home_win_perc', 'first_downs': 'home_first_downs', 'fumbles': 'home_fumbles', 'fumbles_lost':'home_fumbles_lost', 'interceptions':'home_interceptions', 'net_pass_yards': 'home_net_pass_yards', 'pass_attempts':'home_pass_attempts', 'pass_completions':'home_pass_completions', 'pass_touchdowns':'home_pass_touchdowns', 'pass_yards':'home_pass_yards', 'penalties':'home_penalties', 'points':'home_points', 'rush_attempts':'home_rush_attempts', 'rush_touchdowns':'home_rush_touchdowns', 'rush_yards':'home_rush_yards', 'time_of_possession':'home_time_of_possession', 'times_sacked':'home_times_sacked', 'total_yards':'home_total_yards', 'turnovers':'home_turnovers', 'yards_from_penalties':'home_yards_from_penalties', 'yards_lost_from_sacks': 'home_yards_lost_from_sacks', 'fourth_down_perc':'home_fourth_down_perc', 'third_down_perc':'home_third_down_perc'})         
        agg_weekly_df = pd.merge(away_df,home_df,left_on = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr', 'week'], right_on = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr', 'week'])         
        agg_weekly_df['win_perc_dif'] = agg_weekly_df['away_win_perc'] - agg_weekly_df['home_win_perc']         
        agg_weekly_df['first_downs_dif'] = agg_weekly_df['away_first_downs'] - agg_weekly_df['home_first_downs']         
        agg_weekly_df['fumbles_dif'] = agg_weekly_df['away_fumbles'] - agg_weekly_df['home_fumbles']         
        agg_weekly_df['interceptions_dif'] = agg_weekly_df['away_interceptions'] - agg_weekly_df['home_interceptions']         
        agg_weekly_df['net_pass_yards_dif'] = agg_weekly_df['away_net_pass_yards'] - agg_weekly_df['home_net_pass_yards']         
        agg_weekly_df['pass_attempts_dif'] = agg_weekly_df['away_pass_attempts'] - agg_weekly_df['home_pass_attempts']         
        agg_weekly_df['pass_completions_dif'] = agg_weekly_df['away_pass_completions'] - agg_weekly_df['home_pass_completions']         
        agg_weekly_df['pass_touchdowns_dif'] = agg_weekly_df['away_pass_touchdowns'] - agg_weekly_df['home_pass_touchdowns']         
        agg_weekly_df['pass_yards_dif'] = agg_weekly_df['away_pass_yards'] - agg_weekly_df['home_pass_yards']         
        agg_weekly_df['penalties_dif'] = agg_weekly_df['away_penalties'] - agg_weekly_df['home_penalties']        
        agg_weekly_df['points_dif'] = agg_weekly_df['away_points'] - agg_weekly_df['home_points']         
        agg_weekly_df['rush_attempts_dif'] = agg_weekly_df['away_rush_attempts'] - agg_weekly_df['home_rush_attempts']
        agg_weekly_df['rush_touchdowns_dif'] = agg_weekly_df['away_rush_touchdowns'] - agg_weekly_df['home_rush_touchdowns']
        agg_weekly_df['rush_yards_dif'] = agg_weekly_df['away_rush_yards'] - agg_weekly_df['home_rush_yards']
        agg_weekly_df['time_of_possession_dif'] = agg_weekly_df['away_time_of_possession'] - agg_weekly_df['home_time_of_possession']
        agg_weekly_df['times_sacked_dif'] = agg_weekly_df['away_times_sacked'] - agg_weekly_df['home_times_sacked']
        agg_weekly_df['total_yards_dif'] = agg_weekly_df['away_total_yards'] - agg_weekly_df['home_total_yards']
        agg_weekly_df['turnovers_dif'] = agg_weekly_df['away_turnovers'] - agg_weekly_df['home_turnovers'] 
        agg_weekly_df['yards_from_penalties_dif'] = agg_weekly_df['away_yards_from_penalties'] - agg_weekly_df['home_yards_from_penalties']
        agg_weekly_df['yards_lost_from_sacks_dif'] = agg_weekly_df['away_yards_lost_from_sacks'] - agg_weekly_df['home_yards_lost_from_sacks']
        agg_weekly_df['fourth_down_perc_dif'] = agg_weekly_df['away_fourth_down_perc'] - agg_weekly_df['home_fourth_down_perc']
        agg_weekly_df['third_down_perc_dif'] = agg_weekly_df['away_third_down_perc'] - agg_weekly_df['home_third_down_perc']
        agg_weekly_df = agg_weekly_df.drop(columns = ['away_win_perc', 'away_first_downs', 'away_fumbles', 'away_fumbles_lost', 'away_interceptions','away_net_pass_yards', 'away_pass_attempts','away_pass_completions', 'away_pass_touchdowns', 'away_pass_yards', 'away_penalties', 'away_points', 'away_rush_attempts', 'away_rush_touchdowns', 'away_rush_yards', 'away_time_of_possession', 'away_times_sacked', 'away_total_yards', 'away_turnovers', 'away_yards_from_penalties', 'away_yards_lost_from_sacks','away_fourth_down_perc', 'away_third_down_perc','home_win_perc', 'home_first_downs', 'home_fumbles', 'home_fumbles_lost', 'home_interceptions', 'home_net_pass_yards', 'home_pass_attempts','home_pass_completions', 'home_pass_touchdowns', 'home_pass_yards', 'home_penalties', 'home_points', 'home_rush_attempts', 'home_rush_touchdowns', 'home_rush_yards', 'home_time_of_possession', 'home_times_sacked', 'home_total_yards', 'home_turnovers', 'home_yards_from_penalties', 'home_yards_lost_from_sacks','home_fourth_down_perc', 'home_third_down_perc'])
        if (agg_weekly_df['winning_name'].isnull().values.any() and weeks[w]> 3):
            agg_weekly_df['result'] = np.nan
            print(f"Week {weeks[w]} games have not finished yet.")        
        else:
            agg_weekly_df['result'] = agg_weekly_df['winning_name'] == agg_weekly_df['away_name']
            agg_weekly_df['result'] = agg_weekly_df['result'].astype('float')
            agg_weekly_df = agg_weekly_df.drop(columns = ['winning_name', 'winning_abbr']) 
            agg_games_df = pd.concat([agg_games_df, agg_weekly_df])
            agg_games_df = agg_games_df.reset_index().drop(columns = 'index')
    
    agg_games_df = agg_games_df.drop(index = 20, axis=0)
            
    return agg_games_df
    


def main():
    # Tests for offline storage function
    if(False):
        store_data(2023, 10, 10)

    # Tests for sportsipy_submodule_summary function
    # No Offline Mode Implemented
    if(False):
        sportsipy_submodule_summary()
        
    # Tests for get_schedule function. Week 13 of 2022 has a Tie so its a good query.
    if(False):
        print(get_schedule(2023, 1, 1).to_string())

    # Tests for clean_game_data function
    if(False):
        w = 1
        year = 2023
        game_num = 0
        date_str = str(w) + '-' + str(year)
        if(OFFLINE_MODE):
            # Load and deserialize Boxscores(W, YYYY)
            with open(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\WeekBoxscore\\Boxscores_Wk{w}_{year}.pkl', 'rb') as file: 
                weekXthruY_BOX = pickle.load(file)
        else:
            weekXthruY_BOX = Boxscores(1, year, w)

        # Get game 0's URI and Boxscore
        weekX_gameY_URI = weekXthruY_BOX.games[date_str][game_num]['boxscore']
        if(OFFLINE_MODE):
            weekX_gameY_BOX_DF = pd.read_csv(f'C:\\work\\CIS600-Predicting-NFL-Games\\Data\\GameBoxscore\\Boxscore_{weekX_gameY_URI}.csv')
        else:
            weekX_gameY_BOX_DF = Boxscore(weekX_gameY_URI).dataframe

        # Print brief summary of a single game within Boxscore's scope.
        # This is the short version of a Boxscore Object
        weekX_gameY_SUM = weekXthruY_BOX.games[date_str][game_num]
        weekX_gameY_SUM_DF = pd.DataFrame.from_dict([weekX_gameY_SUM])

        away_STATS_DF, home_STATS_DF = get_clean_game_data(weekX_gameY_SUM_DF, weekX_gameY_BOX_DF)
        print(away_STATS_DF.to_string())
        print(home_STATS_DF.to_string())

    # Tests for get_game_data_for_weeks
    if(True):   
        print(get_game_data_for_weeks([1,2], 2022).to_string())

    # Tests for agg_weekly_data
    if(False):
        #agg_weekly_data(schedule_df,weeks_games_df,current_week,weeks):
        schedule_df = get_schedule(2023, 1, 18)
        weeks_games_df = get_game_data_for_weeks([1, 2, 3], 2023)
        current_week = 4
        weeks = list(range(1,current_week + 1))
        get_game_data_for_weeks(schedule_df, weeks_games_df, current_week, weeks)
        print(get_game_data_for_weeks)
    
if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Exit: Keyboard Interrupted")
        sys.exit(0)

# %%
