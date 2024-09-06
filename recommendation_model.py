import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:10000, :]  # Limit to the first 10,000 rows for efficiency
    df = df[['Id', 'ProductId', 'Score', 'Summary', 'Text']]
    return df


# Function to create user-item matrix
def create_user_item_matrix(df):
    user_item_matrix = df.pivot_table(index='Id', columns='ProductId', values='Score').fillna(0)
    return user_item_matrix


# Function to recommend items based on user similarity
def recommend_items(user_item_matrix, target_user_index, k=10):
    user_similarity = cosine_similarity(user_item_matrix)
    neighbor_indices = user_similarity[target_user_index].argsort()[::-1][1:]

    recommended_items = []
    target_user_rating = user_item_matrix.loc[target_user_index].values

    for neighbor_index in neighbor_indices:
        neighbor_rating = user_item_matrix.iloc[neighbor_index].values

        # Find items that neighbor has rated highly but the target user has not rated
        recommended_indices = [i for i, rating in enumerate(neighbor_rating) if
                               rating > 4 and target_user_rating[i] == 0]

        # Add the recommended items to the list
        recommended_items.extend(user_item_matrix.columns[recommended_indices])

    # Ensure unique recommendations and limit to the top k items
    recommended_items = list(set(recommended_items))[:k]
    return recommended_items


# Function to get recommendations from the DataFrame
def get_recommended_df(df, recommended_items):
    recommended_df = df[df['ProductId'].isin(recommended_items)]
    recommended_df = recommended_df.drop_duplicates(subset=['ProductId'])
    return recommended_df.head(10)


# Main function to run the recommendation system
def main():
    file_path = 'datasets\\reviews.csv'  # Replace with your actual dataset path
    df = load_and_preprocess_data(file_path)

    user_item_matrix = create_user_item_matrix(df)

    target_user_index = 10  # Example user index
    recommended_items = recommend_items(user_item_matrix, target_user_index)

    recommended_df = get_recommended_df(df, recommended_items)

    print("Top Recommendations:")
    print(recommended_df)


# Call the main function when the script is run
if __name__ == '__main__':
    main()
