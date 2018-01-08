

## Method Used

It is a type of unsupervised algorithm which  solves the clustering problem. Its procedure follows a simple and easy  way to classify a given data set through a certain number of  clusters (assume k clusters). Data points inside a cluster are homogeneous and heterogeneous to peer groups.

## Import Libraries


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## Create some Data


```python
from sklearn.datasets import make_blobs # sklearn is a library
```


```python
#if you want to explore sklearn.datasets.make_blobs check the following link
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
#https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.htmlhttp://library.ryerson.ca/
```


```python
# Create Data
data = make_blobs(n_samples=300, n_features=2, 
                           centers=4, cluster_std=1.8, random_state=101)

```


```python
print(data[1]) # by applying the Gaussian Blob function
```

    [3 2 0 3 3 1 0 0 0 2 2 0 0 3 1 2 1 3 0 0 1 1 1 3 0 3 2 3 1 2 0 0 0 0 2 2 3
     0 0 2 1 2 2 3 1 3 1 0 3 0 2 0 2 2 3 1 1 1 3 2 1 3 3 2 1 3 3 0 2 3 1 3 3 1
     1 2 2 2 2 1 3 0 2 1 0 0 0 3 1 1 1 1 1 1 0 2 1 0 3 0 0 1 2 3 3 3 3 1 0 0 0
     2 2 3 1 0 1 0 3 2 1 1 0 3 3 3 3 2 2 3 3 2 3 3 1 0 0 0 3 2 3 2 2 2 1 3 2 2
     3 2 3 0 1 2 2 1 3 1 2 3 1 2 1 0 2 0 0 0 3 0 2 1 1 2 0 0 3 1 2 3 2 1 0 0 0
     2 3 1 0 2 1 2 3 0 2 0 2 3 2 0 2 2 0 2 3 3 3 0 1 3 1 2 0 0 3 1 0 1 2 0 1 2
     2 3 3 3 0 2 3 3 0 0 0 1 0 2 3 1 1 2 3 0 1 3 1 3 0 1 2 0 2 1 1 0 0 0 0 1 3
     1 0 3 1 1 1 3 1 1 1 2 2 0 3 2 2 3 1 3 2 2 0 1 3 1 2 3 3 1 1 1 0 2 1 2 2 1
     0 2 3 0]
    

## Visualize Data


```python
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
#plt.scatter(x from the data,y from the data,c=specifying the group of a datapoint,cmap='rainbow')
```




    <matplotlib.collections.PathCollection at 0x1459c389ac8>




![png](output_9_1.png)


## Learning with KMeans Algorithm from the sklearn.cluster library to create Clusters


```python
from sklearn.cluster import KMeans
```


```python
kmeans = KMeans(n_clusters=4)
#n_clusters --- Number of clusters that we are looking for
#Change number of clusters to see how they change the results
```


```python
kmeans.fit(data[0])
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
kmeans.cluster_centers_
```




    array([[-3.67618674,  7.59124984],
           [-9.51083714, -6.50944858],
           [ 4.18620177,  6.80055146],
           [ 0.22590361,  2.10954545]])




```python
kmeans.labels_
```




    array([0, 2, 3, 0, 0, 1, 3, 3, 3, 2, 3, 3, 3, 0, 1, 2, 1, 0, 3, 3, 1, 1, 1,
           0, 3, 0, 2, 0, 1, 2, 3, 3, 3, 3, 2, 2, 0, 3, 3, 2, 1, 2, 2, 0, 1, 0,
           1, 3, 0, 3, 2, 3, 2, 2, 0, 1, 1, 1, 0, 2, 1, 0, 0, 2, 1, 0, 0, 3, 3,
           0, 1, 0, 3, 1, 1, 2, 2, 3, 2, 1, 0, 3, 3, 1, 3, 3, 3, 3, 1, 1, 1, 1,
           1, 1, 3, 2, 1, 3, 0, 3, 3, 1, 3, 0, 0, 0, 0, 1, 3, 3, 3, 2, 3, 0, 1,
           3, 1, 3, 0, 0, 1, 1, 3, 0, 3, 0, 0, 2, 2, 0, 0, 2, 0, 0, 1, 3, 3, 3,
           0, 2, 0, 2, 2, 2, 1, 0, 2, 2, 0, 2, 0, 3, 1, 2, 2, 1, 0, 1, 2, 0, 1,
           2, 1, 3, 2, 3, 3, 3, 0, 0, 2, 1, 1, 2, 3, 3, 0, 1, 2, 0, 2, 1, 3, 3,
           3, 2, 0, 1, 3, 2, 1, 2, 0, 3, 2, 3, 2, 0, 2, 3, 2, 2, 3, 2, 0, 0, 0,
           3, 1, 0, 1, 3, 3, 3, 0, 1, 3, 1, 2, 3, 1, 2, 2, 0, 0, 0, 3, 2, 0, 0,
           3, 3, 3, 1, 3, 2, 0, 1, 1, 2, 0, 3, 1, 0, 1, 0, 3, 1, 2, 3, 2, 1, 1,
           2, 3, 3, 3, 1, 0, 1, 3, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 3, 0, 2, 2, 0,
           1, 0, 2, 2, 3, 1, 0, 1, 2, 3, 0, 1, 1, 1, 3, 2, 1, 2, 2, 1, 3, 2, 0,
           3])




```python
plt.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
plt.title("K Means")
```




    Text(0.5,1,'K Means')




![png](output_16_1.png)



```python
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.title("Original")
```




    Text(0.5,1,'Original')




![png](output_17_1.png)



```python
#To visualise real clusters and idetified clusters, plot them together
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
```




    <matplotlib.collections.PathCollection at 0x1459c312e80>




![png](output_18_1.png)


Please note that the colors are meaningless in reference between the two plots.

### Learning without using the sklearn.cluster library


```python
## Initialisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})


np.random.seed(200)
k = 3
# centroids[i] = [x, y]
centroids = {
    i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
    for i in range(k)
}
#print randomly generated centeroids
print(centroids)    
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k')
#colmap = {1: 'r', 2: 'g', 3: 'b'}
#for i in centroids.keys():
#    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
```

    {1: [26, 16], 2: [68, 42], 3: [55, 76]}
    


![png](output_21_1.png)



```python
## Initialisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})


np.random.seed(200)
k = 3
# centroids[i] = [x, y]
centroids = {
    i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
    for i in range(k)
}
#print randomly generated centeroids
print(centroids)    
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
```

    {1: [26, 16], 2: [68, 42], 3: [55, 76]}
    


![png](output_22_1.png)



```python
## Assignment Stage

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df.head())

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
```

        x   y  distance_from_1  distance_from_2  distance_from_3  closest color
    0  12  39        26.925824        56.080300        56.727418        1     r
    1  20  36        20.880613        48.373546        53.150729        1     r
    2  28  30        14.142136        41.761226        53.338541        1     r
    3  18  52        36.878178        50.990195        44.102154        1     r
    4  29  54        38.118237        40.804412        34.058773        3     b
    


![png](output_23_1.png)



```python
## Update Stage

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()
```


![png](output_24_0.png)



```python
## Repeat Assigment Stage

df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
```


![png](output_25_0.png)


Note that one of the reds is now green and one of the blues is now red.

We are getting closer.

We now repeat until there are no changes to any of the clusters.


```python
# Continue until all assigned categories don't change any more
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
```


![png](output_27_0.png)


So we have 3 clear clusters with 3 means at the centre of these clusters.
