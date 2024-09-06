import numpy as np
import pandas as pd
import textdistance
from collections import Counter
import re


# Function to load the data and prepare word frequency
def load_data(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = data.lower()  # Convert text to lowercase
        words = re.findall(r'\w+', data)  # Extract words
    return words


# Function to calculate word frequencies
def calculate_word_frequencies(words):
    word_freq_dict = Counter(words)
    total_words_freq = sum(word_freq_dict.values())
    return word_freq_dict, total_words_freq


# Function to calculate probabilities of words
def calculate_probabilities(word_freq_dict, total_words_freq):
    probs = {}
    for k in word_freq_dict.keys():
        probs[k] = word_freq_dict[k] / total_words_freq
    return probs


# Autocorrect function
def autocorrect(word, probs, word_freq_dict):
    word = word.lower()
    if word in probs:
        return f'The word "{word}" is already there.'

    similarities = [1 - textdistance.Jaccard(qval=2).distance(w, word) for w in word_freq_dict.keys()]

    df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
    df = df.rename(columns={'index': 'Word', 0: 'Prob'})
    df['Similarity'] = similarities

    # Ensure 'Prob' column is correctly created
    if 'Prob' not in df.columns:
        return "Probability column is missing."

    output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(10)
    return output


# Main execution function
def main():
    # Load words from the book
    words = load_data(r'datasets/autocorrect book.txt')  # Use raw string notation or replace with forward slashes

    # Calculate word frequencies and total word count
    word_freq_dict, total_words_freq = calculate_word_frequencies(words)

    # Calculate the probability of each word
    probs = calculate_probabilities(word_freq_dict, total_words_freq)

    # Get user input
    user_input = input("Enter a word to autocorrect: ")

    # Test the autocorrect function
    result = autocorrect(user_input, probs, word_freq_dict)
    if result is not None:
        print("Top suggestions:")
        print(result)


# Call main function when the script is run
if __name__ == '__main__':
    main()
