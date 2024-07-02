import numpy as np
import pandas as pd

matches = pd.read_csv('IPLMatches.csv')
deliveries = pd.read_csv('deliveries.csv')

# Obtaining total scores from first and second innings of each game into a dataframe
total_score_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()

# Only storing scores from 1st innings (so that the runs required to chase are stored)
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Creates a new data frame which combines the data from matches.csv with their first innings scores
match_df = matches.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

teams = [
    'Sunrisers Hyderabad',
    'Kolkata Knight Riders',
    'Chennai Super Kings',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kings XI Punjab',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Replacing names of teams that have changed names
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

# Ensuring the dataframe only consists of teams which are in the teams list and ignores teams such as Kochi Tuskers Kerala etc
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# Removing all matches that were impacted by rain and hence DLS wasn't applied
match_df = match_df[match_df['dl_applied'] == 0]

# Keeping relevant table headings, and then merging with the deliveries section to get ball by ball data
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
delivery_df = match_df.merge(deliveries, on='match_id')

# Storing the deliveries of all deliveries in a run chase
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Ensure total_runs_y is numeric
delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce').fillna(0)

# Getting total runs at each ball
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

# Getting the total runs left at each ball
delivery_df['runs_left'] = delivery_df['total_runs_x'] + 1 - delivery_df['current_score']

# Getting the balls left after each ball
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

# We are trying to implement a column to represent the number of wickets left

# We start by replacing all the NaN values with 0
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")

# If the batsman's name isn't present i.e. NaN, put "0", else "1"
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else "1")

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

# combining all the wickets to get the total
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets

# current run rate = runs/overs
delivery_df['current_run_rate'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
print(delivery_df)
