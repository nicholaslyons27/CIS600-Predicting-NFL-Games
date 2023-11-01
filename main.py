# Code Comment below (# %%) wraps file in an Jypter code cell and runs in interactive window. Use for visualization of dataframes
# %%
from sportsipy.nfl.boxscore import Boxscores  # Retrieve a dictionary which contains a major game data of all games being played on a particular day.
from sportsipy.nfl.boxscore import Boxscore  # Detailed information about the final statistics for a game.
import pandas as pd
import sys


def main():

    # Create Boxscores Object for the 2023 season weeks 1 - 8
    week1thru8_BOX = Boxscores(1, 2023, 8)
    print("Boxscore Class: ")
    print(week1thru8_BOX)

    # Print dictionary of all games played within the Boxscores' scope (Weeks 1 - 8)
    # Format {Week: [{Array of dictionaries that contain game info}]}
    print("\nGames: ")
    print(week1thru8_BOX.games)

    # Get week 1, game 1's URI
    week1_game1_URI = week1thru8_BOX.games['1-2023'][0]['boxscore']
    print("\nGame 0 URI: ")
    print(week1_game1_URI)

    # Create Detailed Boxscore object using URI
    week1_game1_BOX = Boxscore(week1_game1_URI)

    # Create dataframe out of week 1, game 1's boxscore
    week1_game1_DF = week1_game1_BOX.dataframe

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Exit: Keyboard Interrupted")
        sys.exit(0)

# %%
