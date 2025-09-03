import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df_combined = pd.read_csv("D:/fake-news-detector/data/sample.csv")

# Drop missing titles or labels just in case
df_combined = df_combined.dropna(subset=['title', 'label'])

# TF-IDF on titles
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df_combined['title'])
y = df_combined['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensemble model
lr = LogisticRegression(max_iter=1000)
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=100)

ensemble_model = VotingClassifier(
    estimators=[('lr', lr), ('nb', nb), ('rf', rf)],
    voting='soft'
)

# Train
ensemble_model.fit(X_train, y_train)

# Evaluate
y_pred = ensemble_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
with open(r"D:\fake-news-detector\models\ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)

with open(r"D:\fake-news-detector\models\tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
