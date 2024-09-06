from flask import Flask, render_template, request
import pickle
import pandas as pd
from scipy.stats import poisson
import textdistance
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pickled files if they exist
try:
    with open('dict_table.pkl', 'rb') as f:
        dict_table = pickle.load(f)
except FileNotFoundError:
    dict_table = {}

try:
    with open('word_freq_dict.pkl', 'rb') as f:
        word_freq_dict = pickle.load(f)
    with open('probs.pkl', 'rb') as f:
        probs = pickle.load(f)
except FileNotFoundError:
    word_freq_dict = {}
    probs = {}

try:
    with open('user_item_matrix.pkl', 'rb') as f:
        user_item_matrix = pickle.load(f)
except FileNotFoundError:
    user_item_matrix = pd.DataFrame()  # Placeholder for user-item matrix

# Route for the main page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           football_result='',
                           autocorrect_suggestions=[],
                           recommendations=[],
                           sentiment_result='',
                           user_recommendations=[])

# Route for football data analysis
@app.route('/football', methods=['POST'])
def football():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    result = predict_points(home_team, away_team)
    return render_template('index.html',
                           football_result=f'Home Team Points: {result[0]}, Away Team Points: {result[1]}',
                           autocorrect_suggestions=[],
                           recommendations=[],
                           sentiment_result='',
                           user_recommendations=[])

def predict_points(home, away):
    if home in df_team_strength.index and away in df_team_strength.index:
        lamb_home = df_team_strength.at[home, 'GoalsScored'] * df_team_strength.at[away, 'GoalsConceded']
        lamb_away = df_team_strength.at[away, 'GoalsScored'] * df_team_strength.at[home, 'GoalsConceded']
        prob_home, prob_away, prob_draw = 0, 0, 0
        for x in range(0, 11):
            for y in range(0, 11):
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

# Route for autocorrect
@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    word = request.form['word']
    suggestions = get_autocorrect_suggestions(word)
    return render_template('index.html',
                           football_result='',
                           autocorrect_suggestions=[s[0] for s in suggestions],
                           recommendations=[],
                           sentiment_result='',
                           user_recommendations=[])

def get_autocorrect_suggestions(word):
    if word in probs:
        return [(word, probs[word])]
    else:
        similarities = [1 - (textdistance.Jaccard(qval=2)).distance(w, word) for w in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(10)
        return output.values.tolist()

# Route for rating recommendations
@app.route('/recommendations', methods=['POST'])
def rating_recommendations():
    user_id = int(request.form['user_id'])
    recommendations = get_rating_recommendations(user_id)
    return render_template('index.html',
                           football_result='',
                           autocorrect_suggestions=[],
                           recommendations=recommendations,
                           sentiment_result='',
                           user_recommendations=[])

def get_rating_recommendations(user_id):
    user_ratings = user_item_matrix.loc[user_id, :].values.reshape(1, -1)
    user_similarity = cosine_similarity(user_ratings, user_item_matrix)
    neighbor_indices = user_similarity[0].argsort()[::-1][1:]
    recommended_items = []
    for neighbor_index in neighbor_indices:
        neighbor_rating = user_item_matrix.iloc[neighbor_index]
        recommended_indices = [i for i, rating in enumerate(neighbor_rating) if rating > 4 and user_item_matrix.iloc[user_id, i] == 0]
        recommended_items.extend(user_item_matrix.columns[recommended_indices])
    return [{'ProductId': item, 'Score': user_item_matrix[item].mean()} for item in recommended_items]

# Route for sentiment analysis
@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    text = request.form['text']
    sentiment = analyze_sentiment(text)
    return render_template('index.html',
                           football_result='',
                           autocorrect_suggestions=[],
                           recommendations=[],
                           sentiment_result=sentiment,
                           user_recommendations=[])

def analyze_sentiment(text):
    # Placeholder function, replace with actual sentiment analysis
    return "Positive" if "good" in text.lower() else "Negative"

# Route for user-based recommendations
@app.route('/user_recommendations', methods=['POST'])
def user_based_recommendations():
    user_id = int(request.form['user_id_recommend'])
    recommendations = get_user_based_recommendations(user_id)
    return render_template('index.html',
                           football_result='',
                           autocorrect_suggestions=[],
                           recommendations=[],
                           sentiment_result='',
                           user_recommendations=recommendations)

def get_user_based_recommendations(user_id):
    user_ratings = user_item_matrix.loc[user_id, :].values.reshape(1, -1)
    user_similarity = cosine_similarity(user_ratings, user_item_matrix)
    similar_item_indices = user_similarity.argsort()[0, ::-1][:5]
    return [{'ProductId': item, 'Score': user_item_matrix[item].mean()} for item in similar_item_indices]

if __name__ == '__main__':
    app.run(debug=True)
