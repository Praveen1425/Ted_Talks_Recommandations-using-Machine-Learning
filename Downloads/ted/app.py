from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_url_path='', static_folder='.')

# Load pre-trained model and preprocessed data
try:
    with open("ted_talks_recommendation_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        vectorizer = model_data['tfidf_vectorizer']
        df = model_data['recommendation_data']
except FileNotFoundError:
    print("Error: 'ted_talks_recommendation_model.pkl' not found. Please run main.py first.")
    exit()

# Load the original dataset to get the full talk titles and details
try:
    original_df = pd.read_csv("tedx_dataset.csv")
except FileNotFoundError:
    print("Error: 'tedx_dataset.csv' not found. Please ensure it is in the same directory.")
    exit()

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    speaker_name = data.get("speaker_name")

    if not speaker_name:
        return jsonify({"error": "No speaker name provided."}), 400

    # Find talks by that speaker
    speaker_talks = df[df["main_speaker"].str.lower() == speaker_name.lower()]
    if speaker_talks.empty:
        return jsonify({"message": f"No talks found for '{speaker_name}'."}), 404

    # Get index of first talk by the speaker
    idx = speaker_talks.index[0]

    # Vectorize all talks and compute similarity
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))

    # Sort and get top 5 recommendations (excluding self)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    results = []
    for i, score in sim_scores:
        # Correctly get the original title from the original_df using the correct index
        original_title = original_df.loc[df.index[i], 'title']
        results.append({
            "main_speaker": df.loc[i, "main_speaker"],
            "details": original_title + " (" + str(round(score * 100, 2)) + "% match)"
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)