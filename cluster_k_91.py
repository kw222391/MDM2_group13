import os
import random
import numpy as np
import pandas as pd
import chardet
import re
import io
import pickle
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

    data = data.iloc[:, :3]
    data.columns = ['Breed', 'Description', 'Temperament']
    data['Combined_Text'] = data['Description'].fillna('') + " " + data['Temperament'].fillna('')
    data['Cleaned_Text'] = data['Combined_Text'].apply(preprocess_text)
    data['Tokenized_Text'] = data['Cleaned_Text'].apply(tokenize_text)

    print("Data preview:", flush=True)
    print(data.head(), flush=True)
    return data

def train_word2vec_model(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4, epochs=100):
    model = Word2Vec(sentences=tokenized_texts,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     seed=42)
    model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=epochs)
    return model

def compute_tfidf_weights(tokenized_texts):
    documents = [" ".join(tokens) for tokens in tokenized_texts]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_vectorizer, tfidf_matrix, feature_names

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

def get_all_document_vectors(model, tokenized_texts, tfidf_vectorizer, tfidf_matrix, feature_names):
    doc_vectors = []
    for i, tokens in enumerate(tokenized_texts):
        doc_tfidf = tfidf_matrix[i]
        doc_vector = get_weighted_document_vector(model, tokens, tfidf_vectorizer, feature_names, doc_tfidf)
        doc_vectors.append(doc_vector)
    return np.array(doc_vectors)

def tune_cluster_91(X, data, min_cluster_size=3, max_cluster_size=7, label="Word2Vec_TFIDF"):
    n_clusters = 91
    print(f"\nTuning clustering with fixed {n_clusters} clusters...", flush=True)
    kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=min_cluster_size,
                               size_max=max_cluster_size, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    data_copy = data.copy()
    data_copy["Cluster_KMeans_" + label] = cluster_labels
    print_clusters_info(data_copy, "Cluster_KMeans_" + label)
    return data_copy, kmeans

def save_cluster_model(data, model, filename="cluster91.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((data, model), f)
    print(f"Saved clustering model and data to {filename}", flush=True)

def load_cluster_model(filename="cluster91.pkl"):
    with open(filename, "rb") as f:
        data, model = pickle.load(f)
    print(f"Loaded clustering model and data from {filename}", flush=True)
    return data, model

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

def pre_tune_and_save():
    file_path = '1.csv'
    data = load_and_prepare_data(file_path)
    if data is None:
        return None, None, None, None
    tokenized_texts = data['Tokenized_Text'].tolist()
    w2v_model = train_word2vec_model(tokenized_texts)
    tfidf_vectorizer, tfidf_matrix, feature_names = compute_tfidf_weights(tokenized_texts)
    doc_vectors = get_all_document_vectors(w2v_model, tokenized_texts, tfidf_vectorizer, tfidf_matrix, feature_names)
    tuned_data, kmeans_model = tune_cluster_91(doc_vectors, data)
    save_cluster_model(tuned_data, kmeans_model)
    return tuned_data, kmeans_model, tfidf_vectorizer, (w2v_model, feature_names)

def main():
    if os.path.exists("cluster91.pkl"):
        best_data, best_model = load_cluster_model()
    else:
        best_data, best_model, tfidf_vectorizer, aux = pre_tune_and_save()
        if best_data is None:
            return

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

    print("\nEnter dog characteristics:", flush=True)
    user_input = input("Enter dog characteristics: ")
    predict_cluster(user_input, tfidf_vectorizer, w2v_model, feature_names, best_model, best_data)

if __name__ == "__main__":
    main()
