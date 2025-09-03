import pickle
import sys

# Load model and vectorizer
with open("D:/fake-news-detector/models/ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("D:/fake-news-detector/models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def predict_title(title: str):
    X = tfidf.transform([title])
    pred = model.predict(X)[0]
    return "REAL" if pred == 1 else "FAKE"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'Your news headline here'")
    else:
        title = sys.argv[1]
        print("Prediction:", predict_title(title))
