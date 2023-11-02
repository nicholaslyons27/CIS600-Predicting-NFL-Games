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
        
        _DF                     Any object/variable coverted to a pandas.DataFrame           
"""

def sportsipy_submodule_summary():
    """
    Instantiate and print some of the various Data Types within the sportsipy submodule.
    This function is only used for debugging and helping developers understand the each module.

    Args:
        None

    Returns:
        None
    """

    # Create Boxscores Object for the 2023 season weeks 1 - 8
    week1thru8_BOX = Boxscores(1, 2023, 8)
    print("Boxscore Class: ")
    print(week1thru8_BOX)

    # Print dictionary of all games played within the Boxscores' scope (Weeks 1 - 8)
    # Format {Week: [{Array of dictionaries that contain game info}]}
    week1thru8_SUM = week1thru8_BOX.games
    print("\nBoxscore.games: ")
    print(week1thru8_SUM)

    # Print brief summary of a single game within Boxscore's scope.
    # This is the short version of a Boxscore Object
    week1_game1_SUM = week1thru8_BOX.games['1-2023'][0]
    week1_game1_SUM_DF = pd.DataFrame.from_dict([week1_game1_SUM])
    print("\nGame 0 Summary: ")
    print(week1_game1_SUM_DF.to_string())

    # Get week 1, game 1's URI
    week1_game1_URI = week1thru8_BOX.games['1-2023'][0]['boxscore']
    print("\nGame 0 URI: ")
    print(week1_game1_URI)

    # Create Detailed Boxscore object using URI
    week1_game1_BOX = Boxscore(week1_game1_URI)
    print("\nBoxscore Week 1 Game 1: ")
    print(week1_game1_URI)

    # Create dataframe out of week 1, game 1's boxscore
    week1_game1_BOX_DF = week1_game1_BOX.dataframe
    print("\Boxscore Week 1 Game 1 DataFrame: ")
    print(week1_game1_BOX_DF.to_string())

def get_schedule(year, firstweek, lastweek):
    """
    Create a pandas.DataFrame of games played within a specified timeframe.

    Args:
        year (int): Year of schedule query
        firstweek (int): Starting week of schedule query (inclusive)
        lastweek (int): Ending week of schedule query (inclusive)

    Returns:
        pandas.DataFrame: A DataFrame of all games played within the scope of the query, 
                            where each row corresponds to a single game.
    """

    # List of week range. Note that lastweek is not inclusive so we add 1
    weeks_list = list(range(firstweek, lastweek + 1))

    # Instantiate schedule dataframe
    schedule_DF = pd.DataFrame()

    # For each week of the season
    # for w in range(len(weeks_list)):
    for w in weeks_list:

        # Create key in the string format "Week-Year"
        date_str = str(w) + '-' + str(year)

        # Create Boxscores Object for current week w     
        week_w_BOX = Boxscores(w, year)

        # Instantiate dataframe for current week w
        week_games_DF = pd.DataFrame()

        # For each game data dictionary
        for g in range(len(week_w_BOX.games[date_str])):
            # Create dataframe out of select game statistic keys
            game_DF = pd.DataFrame(week_w_BOX.games[date_str][g], index=[0], columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr'])

            # Add week # to each index
            game_DF['week'] = w

            # Concat current game to list of this weeks game
            week_games_DF = pd.concat([week_games_DF, game_DF])

        # Concat current game to season long dataframe
        schedule_DF = pd.concat([schedule_DF, week_games_DF]).reset_index().drop(columns='index')

    return schedule_DF

def main():
    sportsipy_submodule_summary()

    # Tests for get_schedule function
    # display(get_schedule(2023, 1, 3))
    # print(get_schedule(2023, 1, 1).to_string())

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Exit: Keyboard Interrupted")
        sys.exit(0)

# %%
