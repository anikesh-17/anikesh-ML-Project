import pandas as pd
import pickle
from scipy.stats import poisson

# Load the necessary files
dict_table = pickle.load(open('datasets\dict_table_football', 'rb'))
df_historical_data = pd.read_csv('datasets\clean_fifa_worldcup_matches.csv')
df_fixture = pd.read_csv('datasets\clean_fifa_worldcup_fixture.csv')

# Data preparation for team strengths
df_home = df_historical_data[['HomeTeam', 'HomeGoals', 'AwayGoals']]
df_away = df_historical_data[['AwayTeam', 'HomeGoals', 'AwayGoals']]

df_home = df_home.rename(columns={'HomeTeam': 'Team', 'HomeGoals': 'GoalsScored', 'AwayGoals': 'GoalsConceded'})
df_away = df_away.rename(columns={'AwayTeam': 'Team', 'HomeGoals': 'GoalsConceded', 'AwayGoals': 'GoalsScored'})

df_team_strength = pd.concat([df_home, df_away], ignore_index=True).groupby(['Team']).mean()

# Prediction of match outcome based on Poisson distribution
def predict_points(home, away):
    if home in df_team_strength.index and away in df_team_strength.index:
        lamb_home = df_team_strength.at[home, 'GoalsScored'] * df_team_strength.at[away, 'GoalsConceded']
        lamb_away = df_team_strength.at[away, 'GoalsScored'] * df_team_strength.at[home, 'GoalsConceded']

        prob_home, prob_away, prob_draw = 0, 0, 0
        for x in range(0, 11):  # Home team goals
            for y in range(0, 11):  # Away team goals
                p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                if x == y:
                    prob_draw += p
                elif x > y:
                    prob_home += p
                else:
                    prob_away += p

        points_home = 3 * prob_home + prob_draw
        points_away = 3 * prob_away + prob_draw
        return (points_home, points_away)
    else:
        return (0, 0)

# Example usage
print(predict_points('England', 'United States'))
print(predict_points('Argentina', 'Mexico'))
print(predict_points('Qatar (H)', 'Ecuador'))  # Example for no points if unknown team

# Fixture management and knockout rounds
df_fixture_group_48 = df_fixture[:48].copy()
df_fixture_knockout = df_fixture[48:56].copy()
df_fixture_quarter = df_fixture[56:60].copy()
df_fixture_semi = df_fixture[60:62].copy()
df_fixture_final = df_fixture[62:].copy()

# Update group points based on predicted match outcomes
for group in dict_table:
    teams_in_group = dict_table[group]['Team'].values
    df_fixture_group_6 = df_fixture_group_48[df_fixture_group_48['home'].isin(teams_in_group)]
    for index, row in df_fixture_group_6.iterrows():
        home, away = row['home'], row['away']
        points_home, points_away = predict_points(home, away)

        # Cast points to integer before adding to the table
        dict_table[group].loc[dict_table[group]['Team'] == home, 'Pts'] += int(round(points_home))
        dict_table[group].loc[dict_table[group]['Team'] == away, 'Pts'] += int(round(points_away))

    dict_table[group] = dict_table[group].sort_values('Pts', ascending=False).reset_index(drop=True)
    dict_table[group] = dict_table[group][['Team', 'Pts']].round(0)


# Assign group winners and runners-up to knockout stages
for group in dict_table:
    group_winner = dict_table[group].loc[0, 'Team']
    runners_up = dict_table[group].loc[1, 'Team']
    df_fixture_knockout.replace({f'Winners {group}': group_winner, f'Runners-up {group}': runners_up}, inplace=True)

df_fixture_knockout['winner'] = '?'

# Function to predict winners in knockout stages
def get_winner(df_fixture_updated):
    for index, row in df_fixture_updated.iterrows():
        home, away = row['home'], row['away']
        points_home, points_away = predict_points(home, away)
        winner = home if points_home > points_away else away
        df_fixture_updated.loc[index, 'winner'] = winner
    return df_fixture_updated

get_winner(df_fixture_knockout)

# Function to update the table for knockout rounds
def update_table(df_fixture_round_1, df_fixture_round_2):
    for index, row in df_fixture_round_1.iterrows():
        winner = df_fixture_round_1.loc[index, 'winner']
        match = df_fixture_round_1.loc[index, 'score']
        df_fixture_round_2.replace({f'Winners {match}': winner}, inplace=True)
    df_fixture_round_2['winner'] = '?'
    return df_fixture_round_2

# Perform updates and predict winners in subsequent rounds
update_table(df_fixture_knockout, df_fixture_quarter)
get_winner(df_fixture_quarter)

update_table(df_fixture_quarter, df_fixture_semi)
get_winner(df_fixture_semi)

update_table(df_fixture_semi, df_fixture_final)
get_winner(df_fixture_final)

