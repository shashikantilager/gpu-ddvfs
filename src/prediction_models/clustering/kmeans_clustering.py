import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("feature_set_normalisedavg_default_clock_ctg_converted_with_appname.csv")

independent_variables = open("../filtered_independent_variables.txt").read().splitlines()

categorical_features = open("../excluded_categorical_columns.txt").read().splitlines()
independent_variables += categorical_features

X = pd.DataFrame(df, columns=independent_variables)
print(X.shape)
print(X.columns)
#
# scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
################ Cluster analysis
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.xlabel('Number of clusters')
plt.ylabel('Error (sum of square)')
plt.savefig("kmeanselbowplot.pdf")
plt.show()

kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
# print("Shape: X {} {}".format(X.shape, X.c))
# joblib.dump(kmeans, "Kmeans_cluster_5.pkl")

clusterlabels = kmeans.labels_

df['kmeans_cluster_labels'] = clusterlabels
df.to_csv('kmeans_cluster_label_data2.csv', index=False)
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()

# zero= kmeans.predict(np.array(X.iloc[1]).reshape(1,-1))


# 4 samples/observations and 2 variables/features
