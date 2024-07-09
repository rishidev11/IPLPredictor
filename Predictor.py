import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

matches = pd.read_csv('data/IPLMatches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

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

# current run rate = runs/balls left
delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])

# required run rate = runs required/balls left
delivery_df['rrr'] = (delivery_df['runs_left'] * 6)/(delivery_df['balls_left'])

def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

# Creating a new column representing the result of the match
delivery_df['result'] = delivery_df.apply(result, axis=1)

# Creating our final dataframe
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result']]

# Shuffles the rows in the dataframe randomly to prevent any bias from forming
final_df = final_df.sample(final_df.shape[0])

final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

# Selects all columns except the last one from final_df and assigns them to X.
# This is typically done to separate the features (input variables) from the target variable.
X = final_df.iloc[:,:-1]

# Selects the last column from final_df and assigns it to y.
# This is typically the target variable that you want to predict.
y = final_df.iloc[:,-1]

# Splits data into training and testing sets
# 0.2 test_size means that 20% of the data will be used for testing, and 80% will be used for training.
# random_state=1: This ensures that the split is reproducible.
# If you run the code multiple times with the same random_state, you will get the same split.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=1)

# A ColumnTransformer is used to apply different transformations to differnet columns of a dataframe
# In this case, OneHotEncoding is applied to the batting_team, bowling_team and city columns
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])], remainder='passthrough')

# Pipeline created to sequentially apply transformations to the data
# First is the ColumnTransformer, next is the LogisticRegression model
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

# Pipeline fit to training data
pipe.fit(X_train, y_train)

# y_pred contains predictions made by the pipeline on the test set X_test.
y_pred = pipe.predict(X_test)


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))

# Tracks progression of a match over each over, using the pipeline to show win/lose probabilities
# Filters the DataFrame x_df for the given match_id and only considers the data at the end of each over (ball == 6).
# Cleans the data and predicts win/lose probabilities for each over using pipe.
# Calculates additional metrics like runs_after_over and wickets_in_over.
# Prints the target score and returns a DataFrame with the progression details and the target score
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[match['ball'] == 6]
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)

    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]

    print("Target-", target)
    temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']]
    return temp_df, target


temp_df,target = match_progression(delivery_df,1,pipe)

plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


joblib.dump(pipe, 'model/cricket_match_predictor.pkl')





