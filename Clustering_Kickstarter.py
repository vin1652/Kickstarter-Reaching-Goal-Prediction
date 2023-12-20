import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score 
from sklearn.cluster import KMeans,DBSCAN 
from sklearn.metrics import silhouette_score 
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
# Import data
kickstarter_df = pd.read_excel("C:/Users/vinay/OneDrive/Documents/.spyder-py3/Kickstarter.xlsx")

# Pre-Processing
kickstarter_df = kickstarter_df.dropna()


# Filtering the DataFrame to keep only rows where 'status' is 'successful' or 'failed'
kickstarter_df = kickstarter_df[kickstarter_df['state'].isin(['successful', 'failed'])]

kickstarter_df['goal_fixed']=kickstarter_df['goal']*kickstarter_df['static_usd_rate']
kickstarter_df = kickstarter_df.drop(columns=['id', 'name','state_changed_at','pledged',
                                               'deadline', 'created_at', 'launched_at','goal','static_usd_rate',
                                               'deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr',
                                               'launched_at_hr','name_len','blurb_len'])
# Pre-Processing: Handling categorical variables
categorical_features = kickstarter_df.select_dtypes(include=['object']).columns
X = pd.get_dummies(kickstarter_df, columns=categorical_features, drop_first=True)

# Using Isolation Forest for detecting outliers
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=5)  # Let the algorithm determine the contamination
iso_forest.fit(X)

# Predict if a data point is an outlier. -1 for outliers, 1 for inliers.
outlier_predictions = iso_forest.predict(X)

# Select all rows that are not outliers
is_inlier = outlier_predictions == 1
X = X[is_inlier]

# Standardize the variables
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to your data and transform it
X_std = scaler.fit_transform(X)

# Convert the scaled data back to a DataFrame and assign column names
X_std = pd.DataFrame(X_std, columns=X.columns)


# Calculate inertia for each value of k
withinss = []
for i in range(2, 12):    
    kmeans = KMeans(n_clusters=i,random_state=5)
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)

# Create a plot
plt.plot(range(2, 12), withinss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Finding optimal K
for i in range (2,12):    
    kmeans = KMeans(n_clusters=i,random_state=5)
    model = kmeans.fit(X_std)
    labels = model.labels_
    print(i,':',silhouette_score(X_std,labels))
 
 # Calculate F-score

# Calculate Calinski-Harabasz Score
for i in range(2, 12):
    kmeans = KMeans(n_clusters=i,random_state=5)
    model = kmeans.fit(X_std)
    labels = model.labels_
    calinski_harabasz_avg = calinski_harabasz_score(X_std, labels)
    print(f'{i} Clusters: Calinski-Harabasz Score = {calinski_harabasz_avg}')


#4 is the optimal number of clusters
kmeans = KMeans(n_clusters=4,random_state=5)
model = kmeans.fit(X_std)
labels = model.labels_
centroids = model.cluster_centers_

# Convert centroids to a DataFrame for easier interpretation
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=X_std.columns)

print("Cluster Centroids:")
print(centroids_df)

# Exporting centroids to a CSV file
centroids_df.to_csv("C:/Users/vinay/OneDrive/Documents/.spyder-py3/cluster_centroids.csv", index=False)
# Calculate the mean of all features across the entire dataset
overall_means = X_std.mean()

# Repeat the overall_means to match the number of clusters
overall_means_aligned = np.tile(overall_means.values, (centroids.shape[0], 1))

# Calculate how much each cluster's centroid deviates from the overall mean
feature_importance = pd.DataFrame(centroids - overall_means_aligned, columns=X_std.columns)

print("Feature Importance for Each Cluster:")
print(feature_importance)
feature_importance.to_csv("C:/Users/vinay/OneDrive/Documents/.spyder-py3/cluster_importance.csv", index=False)

    
 # Parameters for DBSCAN
eps_values = [0.5, 1, 1.5, 2]  # Example values, adjust based on your data
min_samples_values = [5, 10, 15]  # Example values

# Running DBSCAN with different parameters and calculating evaluation metrics
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_std)

        # Evaluate only if more than one cluster is found and there are no noise-only clusters
        if len(set(labels)) > 1 and -1 not in labels:
            silhouette = silhouette_score(X_std, labels)
            calinski_harabasz = calinski_harabasz_score(X_std, labels)
            print(f'DBSCAN with eps={eps} and min_samples={min_samples}: Silhouette={silhouette}, Calinski-Harabasz={calinski_harabasz}')   
