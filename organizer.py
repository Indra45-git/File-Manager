import os
import shutil
import pickle
import mimetypes


def extract_text(file_path):
    """Simple extractor—extend for PDF/DOCX if needed."""
    try:
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def load_model():
    """Load the trained vectorizer and classifier from the pickle file."""
    model_path = "model/classifier.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        vectorizer, model = pickle.load(f)  # assuming you saved (vectorizer, classifier)
    
    return vectorizer, model


def organize_files(source_folder):
    if not os.path.isdir(source_folder):
        print(f"Error: Folder '{source_folder}' does not exist!")
        return

    vectorizer, model = load_model()

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Optional: skip hidden files or the model folder itself
        if filename.startswith('.') or filename == 'model':
            continue

        print(f"Processing: {filename}")
        text = extract_text(file_path)

        if not text.strip():
            print(f"  → No text extracted from {filename}, skipping.")
            continue

        # Predict category
        prediction = model.predict(vectorizer.transform([text]))[0]
        target_folder = os.path.join(source_folder, prediction.capitalize())

        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, filename)

        shutil.move(file_path, target_path)
        print(f"Moved '{filename}' → '{prediction.capitalize()}' folder")


if __name__ == "__main__":  # ← Fixed: double underscores!
    folder = input("Enter folder path: ").strip().strip('"\'')
    organize_files(folder)