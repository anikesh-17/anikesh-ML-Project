import pandas as pd

# Load your dataset (replace 'Train.csv' with the path to your dataset)
data = pd.read_csv('datasets\\sentiment_model_dataset.csv')
print(data.head())


print(data.shape)  # Prints the shape of the dataset
print(data['label'].value_counts())  # Counts the number of positive and negative reviews


data = data.iloc[:10000, :]  # Use only the first 10,000 rows


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
porter = PorterStemmer()

def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub('[\W+]', ' ', text.lower())  # Remove special characters and convert to lowercase
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]  # Stem words
    return ' '.join(text)

# Apply preprocessing to the entire dataset
data['text'] = data['text'].apply(preprocess_text)
print(data['text'].head())


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['text']).toarray()
y = data['label'].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5, max_iter=500, scoring='accuracy', random_state=0)
clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


import pickle

with open('clf.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)


