import pandas as pd
import chardet
import re
import io
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score

# For Word2Vec and stopword removal:
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Download NLTK stopwords if not already present
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def detect_encoding(file_path, num_bytes=10000):
    """
    Detects the encoding of a file by reading a sample of its bytes.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print("Detected encoding:", encoding)
    return encoding


def preprocess_text(text):
    """
    Preprocesses the text by removing punctuation/special characters and converting to lowercase.
    """
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return text.lower()


def load_and_prepare_data(file_path):
    """
    Loads the CSV, selects the first three columns, renames them,
    and preprocesses the text.
    """
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        content = f.read()
    data = pd.read_csv(io.StringIO(content))

    # Select the first three columns and rename them to 'Breed', 'Description', 'Temperament'
    data = data.iloc[:, :3]
    data.columns = ['Breed', 'Description', 'Temperament']

    # Combine text columns and preprocess
    data['Combined_Text'] = data['Description'].fillna('') + " " + data['Temperament'].fillna('')
    data['Cleaned_Text'] = data['Combined_Text'].apply(preprocess_text)

    print("Data preview:")
    print(data.head())
    return data


def vectorize_text_bow(data):
    """
    Vectorizes the cleaned text using bag-of-words (CountVectorizer).
    """
    count_vectorizer = CountVectorizer(stop_words='english')
    X_bow = count_vectorizer.fit_transform(data['Cleaned_Text'])
    print("\nBag-of-Words (CountVectorizer) matrix shape:", X_bow.shape)
    return X_bow


def vectorize_text_tfidf(data):
    """
    Vectorizes the cleaned text using TF-IDF.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(data['Cleaned_Text'])
    print("\nTF-IDF matrix shape:", X_tfidf.shape)
    return X_tfidf


def vectorize_text_word2vec(data, vector_size=100, window=5, min_count=1):
    """
    Tokenizes the cleaned text, removes stopwords, trains a Word2Vec model,
    and returns a document matrix where each document is the average of its word vectors.
    """
    # Tokenize each document and remove stopwords
    tokenized_text = []
    for doc in data['Cleaned_Text']:
        tokens = [word for word in doc.split() if word not in STOP_WORDS]
        tokenized_text.append(tokens)

    # Train Word2Vec on the tokenized corpus
    w2v_model = Word2Vec(sentences=tokenized_text, vector_size=vector_size, window=window,
                         min_count=min_count, workers=4, seed=42)

    # Compute the average vector for each document
    doc_vectors = []
    for tokens in tokenized_text:
        # Collect vectors for tokens present in the model's vocabulary
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            doc_vector = np.mean(vectors, axis=0)
        else:
            doc_vector = np.zeros(vector_size)
        doc_vectors.append(doc_vector)

    X_word2vec = np.vstack(doc_vectors)
    print("\nWord2Vec matrix shape:", X_word2vec.shape)
    return X_word2vec, w2v_model


def kmeans_clustering(X, data, num_clusters, label):
    """
    Clusters the data using KMeans and evaluates using the Davies-Bouldin Index.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    col_name = "Cluster_KMeans_" + label
    data[col_name] = cluster_labels

    # Check if X is sparse; if so, convert to dense
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


def hierarchical_clustering(X, data, num_clusters, label):
    """
    Clusters the data using hierarchical clustering (AgglomerativeClustering)
    and evaluates using the Davies-Bouldin Index.
    """
    agg = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
    try:
        X_dense = X.toarray()
    except AttributeError:
        X_dense = X
    cluster_labels = agg.fit_predict(X_dense)
    col_name = "Cluster_Hier_" + label
    data[col_name] = cluster_labels

    db_index = davies_bouldin_score(X_dense, cluster_labels)
    print(f"\nHierarchical Clustering Davies-Bouldin Index for {label} representation: {db_index}")

    clusters = data.groupby(col_name)['Breed'].apply(list)
    print(f"\nHierarchical Clustering Clusters for {label} representation:")
    for cluster_id, breeds in clusters.items():
        print(f"\nCluster {cluster_id} contains breeds:")
        for breed in breeds:
            print(" -", breed)
    return data


if __name__ == "__main__":
    file_path = '1.csv'  # Ensure this file path is correct
    data = load_and_prepare_data(file_path)

    # Determine the number of clusters (aiming for around 5 breeds per cluster)
    num_breeds = data.shape[0]
    num_clusters = max(1, num_breeds // 5)
    print("Using", num_clusters, "clusters.")

    # --- Using Bag-of-Words Representation ---
    X_bow = vectorize_text_bow(data)
    data = kmeans_clustering(X_bow, data, num_clusters, label="BoW")
    data = hierarchical_clustering(X_bow, data, num_clusters, label="BoW")

    # --- Using TF-IDF Representation ---
    X_tfidf = vectorize_text_tfidf(data)
    data = kmeans_clustering(X_tfidf, data, num_clusters, label="TFIDF")
    data = hierarchical_clustering(X_tfidf, data, num_clusters, label="TFIDF")

    # --- Using Word2Vec Representation ---
    X_w2v, w2v_model = vectorize_text_word2vec(data)
    data = kmeans_clustering(X_w2v, data, num_clusters, label="Word2Vec")
    data = hierarchical_clustering(X_w2v, data, num_clusters, label="Word2Vec")
