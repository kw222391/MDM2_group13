import pandas as pd
import chardet
import re
import io
import numpy as np
import matplotlib.pyplot as plt
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

    # Combine Description and Temperament columns, then preprocess the text.
    data['Combined_Text'] = data['Description'].fillna('') + " " + data['Temperament'].fillna('')
    data['Cleaned_Text'] = data['Combined_Text'].apply(preprocess_text)

    print("Data preview:")
    print(data.head())
    return data

def vectorize_text_tfidf(data):
    """
    Use TfidfVectorizer to convert the text to a TF-IDF feature matrix.
    Adjust ngram_range (to include unigrams, bigrams, and trigrams),
    and set min_df and max_df to filter out words that appear too rarely or too frequently,
    in order to improve the discriminability of features.
    """
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Consider unigrams, bigrams, and trigrams
        min_df=2,            # Ignore words that appear in fewer than 2 documents
        max_df=0.90          # Ignore words that appear in more than 90% of the documents
    )
    X_tfidf = tfidf_vectorizer.fit_transform(data['Cleaned_Text'])
    print("\nTF-IDF matrix shape:", X_tfidf.shape)
    return X_tfidf

def find_optimal_clusters_db_only(X, min_clusters=30, max_clusters=60, step=1):
    """
    Iterate over different numbers of clusters k (from min_clusters to max_clusters)
    and compute only the Davies-Bouldin Index, then plot the index curve to help choose the optimal k.
    Finally, select the optimal k based on the lowest Davies-Bouldin Index.
    """
    results = []
    # If X is a sparse matrix, convert it to a dense matrix for index calculation.
    try:
        X_dense = X.toarray()
    except AttributeError:
        X_dense = X

    for k in range(min_clusters, max_clusters + 1, step):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        db_index = davies_bouldin_score(X_dense, cluster_labels)
        results.append((k, db_index))
        print(f"k={k}: Davies-Bouldin Index = {db_index:.3f}")

    # Plot the index curve
    ks = [r[0] for r in results]
    dbs = [r[1] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(ks, dbs, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Davies-Bouldin Index vs k")
    plt.tight_layout()
    plt.show()

    # Select the optimal k based on the lowest Davies-Bouldin Index
    optimal_k = min(results, key=lambda x: x[1])[0]
    print("Optimal k based on Davies-Bouldin Index:", optimal_k)
    return results, optimal_k

if __name__ == "__main__":
    file_path = '1.csv'  # Ensure the file path is correct.
    data = load_and_prepare_data(file_path)

    # Vectorize the text using TF-IDF
    X_tfidf = vectorize_text_tfidf(data)

    # Search for the optimal number of clusters in the range [30, 60] with a step of 1,
    # calculating only the Davies-Bouldin Index.
    results, optimal_k = find_optimal_clusters_db_only(
        X_tfidf,
        min_clusters=30,
        max_clusters=60,
        step=1
    )

    print("Done.")
