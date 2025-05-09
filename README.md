# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and load dataset.

2. Check for missing values and data info.

3. Use Elbow Method to find the best number of clusters.

4. Apply KMeans with 5 clusters and predict cluster labels.

5. Add cluster labels to the dataset.

6. Visualize the clusters using scatter plot.
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: R Raihaan Ahmed 
RegisterNumber:  212224040543
*/
```
```py
import pandas as pd 
import matplotlib.pyplot as plt 
# Updated path assuming the file is in the current working directory
# If the file is in a different location, adjust the path accordingly
data=pd.read_csv("Mall_Customers.csv") 
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)
y_pred=km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
![image](https://github.com/user-attachments/assets/e51864ef-f7ba-4989-be4f-f83c18841f69)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
