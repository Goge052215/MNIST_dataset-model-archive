import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
import numpy as np

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist["data"], mnist["target"]
y = y.astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce the number of principal components to 100
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

y_train_np = np.array(y_train)

# Euclidean model (L2)
euclidean_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
euclidean_model.fit(X_train_pca, y_train_np)

distances, indices = euclidean_model.kneighbors(X_test_pca)
y_pred_euclidean = mode(y_train_np[indices], axis=1).mode.flatten()

accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
print(f"Test set accuracy with Euclidean model: {accuracy_euclidean:.4f}")
