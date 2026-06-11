import pickle

with open("model/classifier.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

print("Model loaded successfully!")
print(f"Categories the model knows: {model.classes_}")