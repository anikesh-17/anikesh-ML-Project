import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Id', 'ProductId', 'Score', 'Summary', 'Text']]
    return df

# Function to preprocess data and create pivot table
def preprocess_data(df):
    ratings_df = df[['Id', 'ProductId', 'Score']]
    pivot_table = ratings_df.pivot_table(index='Id', columns='ProductId', values='Score', fill_value=0)
    return pivot_table

# Function to calculate item similarity using cosine similarity
def calculate_similarity(pivot_table):
    items_similarity = cosine_similarity(pivot_table)
    return items_similarity

# Function to get top-k recommendations for a given user
def get_recommendations_for_user(user_id, pivot_table, k=5):
    user_ratings = pivot_table.loc[user_id, :].values.reshape(1, -1)
    user_item_similarity = cosine_similarity(user_ratings, pivot_table)
    similar_item_indices = user_item_similarity.argsort()[0, ::-1][:k]
    return similar_item_indices

# Function to filter recommendations based on a minimum score
def recommend_items(ratings_df, min_score=3, top_n=50):
    filtered_recommendations = ratings_df[ratings_df['Score'] >= min_score].head(top_n)
    return filtered_recommendations

# Main function to run the recommendation system
def main():
    # Load the data
    file_path = r'datasets/reviews.csv'  # Replace with your actual dataset path
    df = load_data(file_path)

    # Display some basic information about the data
    print(df.head(3))
    print(f"Shape of the dataset: {df.shape}")

    # Plot the distribution of scores
    df['Score'].value_counts().plot(kind='bar')
    plt.show()

    # Preprocess the data and create a pivot table
    df = df.iloc[:10000, :]  # Limit to the first 10,000 rows for efficiency
    pivot_table = preprocess_data(df)

    # Calculate item similarity
    items_similarity = calculate_similarity(pivot_table)

    # Example: Get top-k recommendations for a given user
    user_id = 4  # Example user ID
    k = 5  # Number of recommendations to get
    similar_item_indices = get_recommendations_for_user(user_id, pivot_table, k)

    # Display the similar items
    print(f"Top {k} recommendations for User {user_id}: {similar_item_indices}")

    # Recommend items based on a minimum score
    recommendations = recommend_items(df[['Id', 'ProductId', 'Score']])

    # Display recommendations
    if not recommendations.empty:
        for index, row in recommendations.iterrows():
            print(f"Product ID: {row['ProductId']}, Score: {row['Score']}")
    else:
        print("No recommendations found.")

# Call the main function when the script is executed
if __name__ == '__main__':
    main()
