import os
import random
import numpy as np
import pandas as pd
import chardet
import re
import io
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

def print_clusters_info(data, cluster_col):
    clusters = data.groupby(cluster_col)['Breed'].apply(list)
    print("\nKMeansConstrained Clusters for representation:", flush=True)
    for cluster_id, breeds in clusters.items():
        print(f"\nCluster {cluster_id} contains breeds:", flush=True)
        for breed in breeds:
            print(" -", breed, flush=True)

def detect_encoding(file_path, num_bytes=10000):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(num_bytes)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print("Detected encoding:", encoding, flush=True)
        return encoding
    except Exception as e:
        print(f"Error detecting encoding for file '{file_path}': {e}", flush=True)
        raise

def preprocess_text(text):
    return re.sub(r"[^A-Za-z0-9\s]", " ", text).lower()

def tokenize_text(text):
    return text.split()

def load_and_prepare_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.", flush=True)
        return None
    try:
        encoding = detect_encoding(file_path)
    except Exception:
        print("Failed to detect encoding. Exiting.", flush=True)
        return None
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", flush=True)
        return None
    try:
        data = pd.read_csv(io.StringIO(content))
    except Exception as e:
        print(f"Error parsing CSV file '{file_path}': {e}", flush=True)
        return None

    # use 3 columns
    data = data.iloc[:, :3]
    data.columns = ['Breed', 'Description', 'Temperament']

    # combine the columns
    data['Combined_Text'] = data['Description'].fillna('') + " " + data['Temperament'].fillna('')
    data['Cleaned_Text'] = data['Combined_Text'].apply(preprocess_text)
    data['Tokenized_Text'] = data['Cleaned_Text'].apply(tokenize_text)

    print("Data preview:", flush=True)
    print(data.head(), flush=True)
    return data

# train word2vec
def train_word2vec_model(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4, epochs=100):
    model = Word2Vec(sentences=tokenized_texts,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     seed=42)
    model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=epochs)
    return model


# tfidf weights
def compute_tfidf_weights(tokenized_texts):
    documents = [" ".join(tokens) for tokens in tokenized_texts]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_vectorizer, tfidf_matrix, feature_names

# tfidf and word2vec
def get_weighted_document_vector(model, tokens, tfidf_vectorizer, feature_names, tfidf_vector):
    indices = tfidf_vector.nonzero()[1]
    weighted_vectors = []
    weights = []
    for idx in indices:
        word = feature_names[idx]
        weight = tfidf_vector[0, idx]
        if word in model.wv.key_to_index:
            weighted_vectors.append(model.wv[word] * weight)
            weights.append(weight)
    if weighted_vectors:
        return np.sum(weighted_vectors, axis=0) / np.sum(weights)
    else:
        return np.zeros(model.vector_size)

# vectorize
def get_all_document_vectors(model, tokenized_texts, tfidf_vectorizer, tfidf_matrix, feature_names):
    doc_vectors = []
    for i, tokens in enumerate(tokenized_texts):
        doc_tfidf = tfidf_matrix[i]
        doc_vector = get_weighted_document_vector(model, tokens, tfidf_vectorizer, feature_names, doc_tfidf)
        doc_vectors.append(doc_vector)
    return np.array(doc_vectors)

def tune_kmeans_clustering(X, data, cluster_range, min_cluster_size=3, max_cluster_size=7, label="Word2Vec_TFIDF"):
    best_db_index = float('inf')
    best_data = None
    best_model = None
    best_num_clusters = None

    # davies-Bouldin index
    db_index_results = {}

    for num_clusters in cluster_range:
        print(f"\nTrying {num_clusters} clusters...", flush=True)
        kmeans = KMeansConstrained(n_clusters=num_clusters, size_min=min_cluster_size,
                                   size_max=max_cluster_size, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        db_index = davies_bouldin_score(X, cluster_labels)
        print(f"random_state=42 - Davies-Bouldin Index: {db_index}", flush=True)
        db_index_results[num_clusters] = db_index

        # find best k
        if db_index < best_db_index:
            best_db_index = db_index
            best_data = data.copy()
            best_data["Cluster_KMeans_" + label] = cluster_labels
            best_model = KMeansConstrained(n_clusters=num_clusters, size_min=min_cluster_size,
                                           size_max=max_cluster_size, random_state=42)

            best_model.cluster_centers_ = np.zeros((num_clusters, X.shape[1]))
            best_num_clusters = num_clusters

    print(f"\nBest Davies-Bouldin Index: {best_db_index} with {best_num_clusters} clusters.", flush=True)
    if best_data is not None:
        print_clusters_info(best_data, "Cluster_KMeans_" + label)
    else:
        print("No valid clustering result was found.", flush=True)
    return best_data, best_model, best_db_index, db_index_results


# plot d
def plot_db_index_chart(db_index_results):
    clusters = sorted(db_index_results.keys())
    db_indices = [db_index_results[k] for k in clusters]

    plt.figure(figsize=(8, 5))
    plt.plot(clusters, db_indices, marker='o', linestyle='-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Finding the best number of Clusters")
    plt.grid(True)
    plt.show()

# Input and output finding the suitable breed
def predict_cluster(user_input, tfidf_vectorizer, w2v_model, feature_names, kmeans_model, data):
    processed_input = preprocess_text(user_input)
    tokens_input = tokenize_text(processed_input)
    tfidf_input = tfidf_vectorizer.transform([processed_input])
    input_vector = get_weighted_document_vector(w2v_model, tokens_input, tfidf_vectorizer, feature_names, tfidf_input)
    centers = kmeans_model.cluster_centers_
    distances = np.linalg.norm(centers - input_vector, axis=1)
    predicted_cluster = np.argmin(distances)
    print(f"\nThe input dog belongs to cluster: {predicted_cluster}", flush=True)
    cluster_data = data[data["Cluster_KMeans_Word2Vec_TFIDF"] == predicted_cluster]
    print("Dogs in the same cluster:", flush=True)
    print(cluster_data["Breed"].tolist(), flush=True)

def main():
    file_path = '1.csv'
    data = load_and_prepare_data(file_path)
    if data is None:
        print("Data loading failed. Please check the file path and format.", flush=True)
        return

    tokenized_texts = data['Tokenized_Text'].tolist()

    print("Training Word2Vec model...", flush=True)
    w2v_model = train_word2vec_model(tokenized_texts, vector_size=100, window=5, min_count=1, workers=1, epochs=100)

    print("Computing TF-IDF weights...", flush=True)
    tfidf_vectorizer, tfidf_matrix, feature_names = compute_tfidf_weights(tokenized_texts)

    print("Computing weighted document vectors...", flush=True)
    doc_vectors = get_all_document_vectors(w2v_model, tokenized_texts, tfidf_vectorizer, tfidf_matrix, feature_names)

    # 277/7 = 40 277/ 30=092.3
    cluster_range = range(40, 92)
    best_data, best_model, best_db_index, db_index_results = tune_kmeans_clustering(
        doc_vectors, data, cluster_range,
        min_cluster_size=3, max_cluster_size=7,
        label="Word2Vec_TFIDF"
    )
    # Plot
    plot_db_index_chart(db_index_results)

    print("DEBUG: Entering interactive input section", flush=True)
    print("\nPlease enter dog characteristics (e.g., description, temperament):", flush=True)
    user_input = input("Enter dog characteristics: ")
    print("DEBUG: User input:", user_input, flush=True)
    predict_cluster(user_input, tfidf_vectorizer, w2v_model, feature_names, best_model, best_data)

if __name__ == "__main__":
    main()
