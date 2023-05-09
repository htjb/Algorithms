import numpy as np
import matplotlib.pyplot as plt
import random

# mock data
x = np.hstack([np.random.normal(0, 0.1, 350), 
               np.random.normal(3, 0.5, 650)])
y = np.hstack([np.random.normal(2, 1, 500), 
               np.random.normal(13, 1.2, 500)])
data = np.vstack([x, y]).T

# call parameters
cluster_number = 3
max_iter = 1000

# inital centroids
means0 = np.random.multivariate_normal(
    [np.mean(x), np.mean(y)], 
    [[np.std(x), 0], [0, np.std(y)]], cluster_number)

count = 0
for iter in range(max_iter):
    if iter == 0:
        means = means0.copy()
    
    # assign points label k based on minimum 
    # euclidean distance to centeroid k
    label = []
    for h in range(len(data)):
        s = []
        for j in range(len(means)):
            s.append(np.sqrt(np.sum(
                [(data[h, k]-means[j, k])**2 
                 for k in range(data.shape[-1])])))
        label.append(np.where(s == min(s))[0][0])
    label = np.array(label)

    # break if labels haven't changed for 
    # 5% of the requested max iters
    # else reset the count
    if iter > 0:
        if np.all(label==old_labels):
            count += 1
            if count > (max_iter/100)*5:
                print('Stopped. Iterations used:' + str(iter) +
                      ' of ' + str(max_iter) + ' requested.')
                break
        else:
            count = 0
    
    # catch to make sure cluster number is respected
    for k in range(cluster_number):
        if not k in set(label):
            label[random.randint(0, len(label)-1)] = k
    
    # readjust centeroids
    for k in range(cluster_number):
        means[k] = np.mean(data[label==k, :], axis=0)
    
    # save current labels
    old_labels = label.copy()

# plot example results
plt.scatter(data[:, 0], data[:, 1], 
            cmap=plt.get_cmap('viridis'), 
            c=label, marker='o', s=10)
plt.scatter(means0[:, 0], means0[:, 1], 
            marker='*', label='Start', 
            cmap=plt.get_cmap('viridis'), 
            c=np.arange(0, cluster_number), 
            ec='k', s=50)
plt.scatter(means[:, 0], means[:, 1], 
            marker='s', label='End', 
            cmap=plt.get_cmap('viridis'), 
            c=np.arange(0, cluster_number), ec='k', s=50)
plt.legend()
plt.savefig('K-means/example_kmeans_clustering.png', dpi=300)
plt.show()