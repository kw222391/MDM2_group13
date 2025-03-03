import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('akc_latest.csv')
df.drop(columns=['Unnamed: 0', 'Unnamed: 13', 'Cluster'], errors='ignore', inplace=True)

df['height_avg'] = (df['min_height'] + df['max_height']) / 2
df['weight_avg'] = (df['min_weight'] + df['max_weight']) / 2
df['expectancy_avg'] = (df['min_expectancy'] + df['max_expectancy']) / 2

features = [
    'grooming_frequency_value', 'shedding_value', 'energy_level_value',
    'trainability_value', 'demeanor_value','weight_avg'
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')


X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow
sse = []
k_range = range(1, 100)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# Silhouette
silhouette_scores = []
k_values = range(2, 200)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k = {k}, Silhouette Score = {score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score ')
plt.show()

#pick k value
k_optimal = 148
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df.loc[X.index, 'Cluster'] = clusters

# plot the graph of clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title(f'K-Means Clustering (k={k_optimal})')
plt.colorbar(label='Cluster Label')
plt.show()

print(df.head())

