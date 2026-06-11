import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_training_data():
    data = []
    labels = []

    # Simple sample data to simulate different file types
    sample_data = {
        "document": [
            "This report contains an analysis of quarterly results",
            "The meeting minutes were recorded and transcribed",
            "Please find attached the contract for review",
            "Executive summary of the annual financial report"
        ],
        "code": [
            "def calculate_sum(a, b):",
            "import numpy as np",
            "public class Main {",
            "function processData(data) {",
            "SELECT * FROM users WHERE active = 1"
        ],
        "image": [
            "PNG image header signature",
            "JFIF JPEG image file",
            "Exif data contains camera model",
            "pixel values RGB color space"
        ],
        "audio": [
            "RIFF WAV file format",
            "ID3 tag metadata MP3",
            "audio samples 44100 Hz",
            "Opus encoded audio stream"
        ],
        "video": [
            "ftypisom MP4 container",
            "WEBM video file header",
            "matroska video container",
            "H.264 video codec stream"
        ]
    }

    for label, texts in sample_data.items():
        for text in texts:
            data.append(text)
            labels.append(label)

    return data, labels


def train_model():
    print("Loading training data...")
    data, labels = load_training_data()

    print(f"Training on {len(data)} samples across {len(set(labels))} categories.")

    # Vectorize text
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(data)

    # Train classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    # Save both vectorizer and model
    os.makedirs("model", exist_ok=True)
    with open("model/classifier.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("Model trained and saved to 'model/classifier.pkl'")


# CORRECT WAY:
if __name__ == "__main__":
    train_model()