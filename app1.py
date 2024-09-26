import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import poisson
import textdistance
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load sentiment analysis model and vectorizer
with open('clf.pkl', 'rb') as model_file:
    sentiment_model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)


# Function for Football Data Analysis
def load_football_data():
    df_historical_data = pd.read_csv('datasets/clean_fifa_worldcup_matches.csv')
    df_fixture = pd.read_csv('datasets/clean_fifa_worldcup_fixture.csv')
    dict_table = pickle.load(open('datasets/dict_table_football', 'rb'))
    return df_historical_data, df_fixture, dict_table


def predict_football(home, away, df_historical_data):
    df_home = df_historical_data[['HomeTeam', 'HomeGoals', 'AwayGoals']]
    df_away = df_historical_data[['AwayTeam', 'HomeGoals', 'AwayGoals']]
    df_home = df_home.rename(columns={'HomeTeam': 'Team', 'HomeGoals': 'GoalsScored', 'AwayGoals': 'GoalsConceded'})
    df_away = df_away.rename(columns={'AwayTeam': 'Team', 'HomeGoals': 'GoalsConceded', 'AwayGoals': 'GoalsScored'})
    df_team_strength = pd.concat([df_home, df_away], ignore_index=True).groupby(['Team']).mean()

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
        return points_home, points_away
    else:
        return 0, 0


# Function for Autocorrect
def load_autocorrect_data():
    with open('datasets/autocorrect_book.txt', 'r', encoding='utf-8') as f:
        data = f.read().lower()
    words = re.findall('\w+', data)
    return Counter(words)


def autocorrect(word, word_freq_dict):
    word = word.lower()
    if word in word_freq_dict:
        return f"The word '{word}' is already correct."
    else:
        similarities = [1 - (textdistance.Jaccard(qval=2)).distance(w, word) for w in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(word_freq_dict, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Freq'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Freq'], ascending=False).head(10)
        return output[['Word', 'Similarity']]


# Function for Rating-based Recommendation
def load_ratings_data():
    df = pd.read_csv('datasets/reviews.csv').iloc[:10000, :]
    pivot_table = df.pivot_table(index='Id', columns='ProductId', values='Score', fill_value=0)
    return pivot_table


def recommend_products(user_id, pivot_table, k=5):
    user_ratings = pivot_table.loc[user_id, :].values.reshape(1, -1)
    user_item_similarity = cosine_similarity(user_ratings, pivot_table)
    similar_item_indices = user_item_similarity.argsort()[0, ::-1][:k]
    return pivot_table.columns[similar_item_indices]


# Function for Sentiment Analysis
def analyze_sentiment(text):
    text_vector = tfidf.transform([text])
    prediction = sentiment_model.predict(text_vector)
    return 'Positive' if prediction[0] == 1 else 'Negative'


# Function for User-based Recommendation
def load_user_data():
    df = pd.read_csv('datasets/reviews.csv').iloc[:10000, :]
    user_item_matrix = df.pivot_table(index='Id', columns='ProductId', values='Score').fillna(0)
    return user_item_matrix


def recommend_user_based(user_id, user_item_matrix):
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(user_id)
    neighbor_indices = user_similarity[target_user_index].argsort()[::-1][1:]
    recommended_items = []
    for neighbor_index in neighbor_indices:
        neighbor_ratings = user_item_matrix.iloc[neighbor_index]
        target_user_ratings = user_item_matrix.iloc[target_user_index]
        recommended_indices = [i for i, rating in enumerate(neighbor_ratings) if
                               rating > 4 and target_user_ratings[i] == 0]
        recommended_items.extend(user_item_matrix.columns[recommended_indices])
    return recommended_items


# Streamlit UI
st.title("5 in 1 Ml Model by Anikesh")

model_selection = st.sidebar.selectbox("Select a Model", [
    "Football Data Analysis",
    "Autocorrect",
    "Rating-based Recommendation",
    "Sentiment Analysis",
    "User-based Recommendation"
])

if model_selection == "Football Data Analysis":
    df_historical_data, df_fixture, dict_table = load_football_data()
    home_team = st.text_input("Home Team")
    away_team = st.text_input("Away Team")
    if st.button("Predict Points"):
        points = predict_football(home_team, away_team, df_historical_data)
        st.write(f"Points Home: {points[0]}, Points Away: {points[1]}")

elif model_selection == "Autocorrect":
    word_freq_dict = load_autocorrect_data()
    word = st.text_input("Word to Correct")
    if st.button("Get Suggestions"):
        suggestions = autocorrect(word, word_freq_dict)
        st.write(suggestions)

elif model_selection == "Rating-based Recommendation":
    pivot_table = load_ratings_data()
    user_id = st.number_input("User ID", min_value=1)
    k = st.number_input("Number of Recommendations", min_value=1, max_value=10)
    if st.button("Get Recommendations"):
        recommendations = recommend_products(user_id, pivot_table, k)
        st.write(recommendations)

elif model_selection == "Sentiment Analysis":
    text = st.text_area("Text to Analyze")
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(text)
        st.write(f"Sentiment: {sentiment}")

elif model_selection == "User-based Recommendation":
    user_item_matrix = load_user_data()
    user_id = st.number_input("User ID", min_value=1)
    if st.button("Get Recommendations"):
        recommendations = recommend_user_based(user_id, user_item_matrix)
        st.write(recommendations)
