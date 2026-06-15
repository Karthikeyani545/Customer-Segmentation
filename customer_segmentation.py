import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

print("Loading dataset...")
df = pd.read_csv("Mall_Customers_India.csv")
print("Dataset Loaded Successfully!")

# Feature arrays set up matching input matrix
X = df[["Annual_Income_LPA", "Spending_Score"]].values

print("Running K-Means algorithm...")
optimal_clusters = 5
kmeans = KMeans(
    n_clusters=optimal_clusters, init="k-means++", random_state=42
)
y_kmeans = kmeans.fit_predict(X)
df["Cluster"] = y_kmeans

# Visualize
plt.figure(figsize=(10, 7))
for i in range(optimal_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f"Cluster {i+1}")

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c="black",
    marker="X",
    label="Centroids",
)
plt.title("Customer Segments (India Market)")
plt.xlabel("Annual Income in LPA (₹)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
print("Displaying output graph...")
plt.show()