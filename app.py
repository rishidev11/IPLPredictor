from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
pipe = joblib.load('model/cricket_match_predictor.pkl')
matches = pd.read_csv('data/IPLMatches.csv')

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        city = request.form['city']
        target = int(request.form['target'])
        current_score = int(request.form['current_score'])
        wickets = int(request.form['wickets'])
        overs_completed = float(request.form['overs_completed'])

        runs_left = target - current_score
        balls_left = 120 - int(overs_completed * 6)


        if (wickets >= 10 and current_score <= target):
            win_probability = 0
            lose_probability = 100
            return render_template('result.html', team1=team1, team2=team2, win_probability=win_probability,lose_probability=lose_probability)


        crr = (current_score * 6) / (overs_completed * 6)
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({
            'batting_team': [team1],
            'bowling_team': [team2],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        win_probability = np.round(result[0][1] * 100, 2)
        lose_probability = np.round(result[0][0] * 100, 2)

        return render_template('result.html', team1=team1, team2=team2, win_probability=win_probability, lose_probability=lose_probability)

    cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
              'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
              'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
              'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
              'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
              'Sharjah', 'Mohali', 'Bengaluru'
    ]


    cities = sorted(cities)

    return render_template('index.html', cities=cities)

if __name__ == '__main__':
    app.run(debug=True)