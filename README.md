# Customer Segmentation Project

I built this project to segment customers based on their demographics and spending behavior using K-Means clustering. The goal was to understand different customer groups so that businesses can create targeted marketing strategies for each segment.

## Tools & Libraries Used
- Python
- pandas, NumPy
- scikit-learn (KMeans, StandardScaler, PCA)
- matplotlib, seaborn

## What I Did
1. **Data Preparation** — Worked with customer data containing Gender, Age, Annual Income, and Spending Score.
2. **Exploratory Data Analysis (EDA)** — Plotted distributions for age, income, spending score, and gender to understand the dataset before clustering.
3. **Preprocessing** — Encoded the Gender column and scaled all numeric features using StandardScaler so no single feature dominates the clustering.
4. **Finding the Right Number of Clusters** — Used the Elbow Method and Silhouette Score to decide how many segments make sense. Ended up with 3 clusters.
5. **Clustering** — Applied K-Means to group customers into segments.
6. **Visualization** — Plotted Income vs Spending Score, Age vs Spending Score, and a PCA 2D projection to visually see the separation between segments.
7. **Cluster Profiling** — Calculated the average age, income, and spending score for each cluster to understand who belongs to each segment.

## Results

| Cluster | Avg Age | Avg Income (k$) | Avg Spending Score | Count |
|---|---|---|---|---|
| 0 | 52.4 | 70.7 | 36.5 | 64 |
| 1 | 25.4 | 45.4 | 74.2 | 40 |
| 2 | 49.1 | 65.7 | 39.9 | 96 |

## Business Insights
- **Cluster 1** (younger, moderate income, high spending) — most responsive to trend-driven and lifestyle marketing.
- **Cluster 0** (older, higher income, lower spending) — good target for premium offers to increase their spend.
- **Cluster 2** (largest group, mid income, moderate-low spending) — the mainstream segment, best suited for value and loyalty campaigns.

## What I Learned
This project helped me understand how unsupervised learning (K-Means) can be used to find hidden patterns in customer data, how to decide the right number of clusters using the elbow method and silhouette score, and how to translate clustering output into actual business insights.
