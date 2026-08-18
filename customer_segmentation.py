# %% [markdown]
# # Customer Segmentation Project
# Segment customers based on behavior and demographics using K-Means clustering.
# Author: Mahek Kamar Pinjari

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

np.random.seed(42)
sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Load / Generate Dataset
# Uses a mall-customer-style dataset: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).
# If you have a real dataset (e.g. Kaggle "Mall Customer Segmentation Data"), just replace this block with:
# `df = pd.read_csv("Mall_Customers.csv")`

# %%
n = 200
genders = np.random.choice(["Male", "Female"], size=n, p=[0.44, 0.56])
age = np.random.randint(18, 70, size=n)

# Build income & spending score with realistic customer segments baked in
income = []
spending = []
for a in age:
    if a < 30:
        inc = np.random.normal(45, 15)
        spend = np.random.normal(70, 15)
    elif a < 45:
        inc = np.random.normal(75, 20)
        spend = np.random.normal(50, 20)
    else:
        inc = np.random.normal(60, 18)
        spend = np.random.normal(35, 18)
    income.append(max(15, round(inc)))
    spending.append(int(np.clip(spend, 1, 100)))

df = pd.DataFrame({
    "CustomerID": range(1, n + 1),
    "Gender": genders,
    "Age": age,
    "Annual_Income_k": income,
    "Spending_Score": spending,
})
print(df.head())
print(df.describe())

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["Age"], bins=15, kde=True, ax=axes[0], color="#0E7C7B")
axes[0].set_title("Age Distribution")
sns.histplot(df["Annual_Income_k"], bins=15, kde=True, ax=axes[1], color="#1B2A4A")
axes[1].set_title("Annual Income Distribution (k$)")
sns.histplot(df["Spending_Score"], bins=15, kde=True, ax=axes[2], color="#D97706")
axes[2].set_title("Spending Score Distribution")
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", data=df, palette=["#0E7C7B", "#D97706"])
plt.title("Gender Split")
plt.tight_layout()
plt.savefig("eda_gender.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Preprocessing
# Encode Gender, scale numeric features so clustering isn't biased by scale.

# %%
le = LabelEncoder()
df["Gender_encoded"] = le.fit_transform(df["Gender"])  # Female=0, Male=1

features = ["Age", "Annual_Income_k", "Spending_Score", "Gender_encoded"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ## 4. Find Optimal Number of Clusters (Elbow Method + Silhouette Score)

# %%
inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o", color="#0E7C7B")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), sil_scores, marker="o", color="#D97706")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Score")
plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.close()

best_k = k_range[int(np.argmax(sil_scores))]
print(f"Best k by silhouette score: {best_k}")

# %% [markdown]
# ## 5. Final Clustering (K-Means)

# %%
K = best_k  # change manually if you want a different k after inspecting the elbow plot
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df["Cluster"].value_counts().sort_index())

# %% [markdown]
# ## 6. Visualize Segments

# %%
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Annual_Income_k", y="Spending_Score",
                 hue="Cluster", palette="Set2", s=70)
plt.title("Customer Segments: Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig("segments_income_spending.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Age", y="Spending_Score",
                 hue="Cluster", palette="Set2", s=70)
plt.title("Customer Segments: Age vs Spending Score")
plt.tight_layout()
plt.savefig("segments_age_spending.png", dpi=150)
plt.close()

# PCA 2D view (useful when using more features than 2)
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = pca_coords[:, 0], pca_coords[:, 1]

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=70)
plt.title("Customer Segments (PCA 2D Projection)")
plt.tight_layout()
plt.savefig("segments_pca.png", dpi=150)
plt.close()

# %% [markdown]
# ## 7. Cluster Profiles (Key Characteristics of Each Segment)

# %%
profile = df.groupby("Cluster")[["Age", "Annual_Income_k", "Spending_Score"]].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
print(profile)

profile.to_csv("cluster_profiles.csv")

plt.figure(figsize=(8, 5))
profile[["Age", "Annual_Income_k", "Spending_Score"]].plot(
    kind="bar", ax=plt.gca(), color=["#0E7C7B", "#1B2A4A", "#D97706"])
plt.title("Average Characteristics per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Value")
plt.tight_layout()
plt.savefig("cluster_characteristics.png", dpi=150)
plt.close()

# %% [markdown]
# ## 8. Save Final Segmented Dataset

# %%
df.to_csv("customers_segmented.csv", index=False)
print("Saved: customers_segmented.csv, cluster_profiles.csv, and all plots (.png)")

# %% [markdown]
# ## 9. Business Insights (write these up in your README/report)
# - Identify which cluster is "high income, high spending" -> premium/loyalty offers
# - Identify "low income, high spending" -> price-sensitive, respond well to discounts
# - Identify "high income, low spending" -> untapped potential, needs targeted marketing
# - Identify "low income, low spending" -> low priority for premium campaigns
# Fill in the actual numbers from your cluster_profiles.csv output.
