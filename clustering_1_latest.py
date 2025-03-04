import pandas as pd
import chardet
import re
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


def detect_encoding(file_path, num_bytes=10000):
    """
    Detect the encoding of the file.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print("Detected encoding:", encoding)
    return encoding


def preprocess_text(text):
    """
    Preprocess the text:
    - Remove punctuation and special characters.
    - Convert to lowercase.
    """
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return text.lower()


def load_and_prepare_data(file_path):
    """
    Load the CSV data, select the first three columns (Breed, Description, Temperament),
    combine the text fields, and preprocess the text.
    """
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        content = f.read()
    data = pd.read_csv(io.StringIO(content))

    # Select the first three columns and rename them.
    data = data.iloc[:, :3]
    data.columns = ['Breed', 'Description', 'Temperament']

    # Combine the Description and Temperament columns, and preprocess the text.
    data['Combined_Text'] = data['Description'].fillna('') + " " + data['Temperament'].fillna('')
    data['Cleaned_Text'] = data['Combined_Text'].apply(preprocess_text)

    print("Data preview:")
    print(data.head())
    return data


def vectorize_text_tfidf(data):
    """
    Convert text to a TF-IDF feature matrix using TfidfVectorizer.
    Adjust ngram_range (to include unigrams, bigrams, and trigrams),
    and set min_df and max_df to filter out words that occur too rarely or too frequently,
    in order to improve feature discriminability.
    """
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 5),  # Consider unigrams, bigrams, and trigrams
        min_df=5,  # Ignore words that appear in fewer than 2 documents
        max_df=0.90  # Ignore words that appear in more than 90% of the documents
    )
    X_tfidf = tfidf_vectorizer.fit_transform(data['Cleaned_Text'])
    print("\nTF-IDF matrix shape:", X_tfidf.shape)
    return X_tfidf


def kmeans_clustering(X, data, num_clusters, label="TFIDF"):
    """
    Perform KMeans clustering on the vectorized data and evaluate the clustering
    using the Davies-Bouldin Index.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    col_name = "Cluster_KMeans_" + label
    data[col_name] = cluster_labels

    # Convert X to a dense matrix if it is sparse.
    try:
        X_dense = X.toarray()
    except AttributeError:
        X_dense = X

    db_index = davies_bouldin_score(X_dense, cluster_labels)
    print(f"\nKMeans Davies-Bouldin Index for {label} representation: {db_index}")

    clusters = data.groupby(col_name)['Breed'].apply(list)
    print(f"\nKMeans Clusters for {label} representation:")
    for cluster_id, breeds in clusters.items():
        print(f"\nCluster {cluster_id} contains breeds:")
        for breed in breeds:
            print(" -", breed)
    return data


if __name__ == "__main__":
    file_path = '1.csv'  # Ensure the file path is correct.
    data = load_and_prepare_data(file_path)

    # Set the number of clusters to 60.
    num_clusters = 60
    print("Using", num_clusters, "clusters.")

    # Vectorize the text using TF-IDF and perform KMeans clustering.
    X_tfidf = vectorize_text_tfidf(data)
    data = kmeans_clustering(X_tfidf, data, num_clusters, label="TFIDF")

    # Save the resulting DataFrame to a CSV file.
    output_file = "clustered_data.csv"
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
