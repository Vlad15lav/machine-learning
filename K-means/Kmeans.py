import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_blobs

class Kmeans:
    def __init__(self, K, iters=100, elbow=False):
        self.__size = K
        self.__iters = iters
        self.__elbow = elbow
        
        self.__centroids = None

    def get(self):
        return {'K': self.__size, 'Centroids': self.__centroids}

    # Spread across clusters
    def __Clustering(self, x_set, u_center):
        N = len(x_set)
        K = len(u_center)
        dists = np.zeros((N, K)) # (X_i, distance to the centroid)
        for i, Nu in enumerate(u_center):
            dists[:, i] = np.linalg.norm(Nu - x_set, axis=1)
        C = np.argmin(dists, axis=1) # number of centroid to X_i

        Get_Index = np.random.choice(np.arange(N), size=K, replace=False)
        for k in range(K): # Check for empty clusters
            if len(C[C == k]) == 0:
                C[Get_Index[k]] = k
        return C

    # Updating centroids
    def __UpdateCenters(self, x_set, clusters, u_center):
        Next_Nu = np.zeros(u_center.shape)
        for k in range(len(u_center)):
            Next_Nu[k] = np.mean(x_set[clusters == k], axis=0)
        return Next_Nu
    
    # Loss
    def __Error(self, x_set, clusters, u_center):
        E = 0
        for k in range(len(u_center)):
            E += np.sum(np.linalg.norm(u_center[k] - x_set[clusters == k], axis=1) ** 2)
        return E
    
    # Best size K-means (Elbow Method)
    def __ElbowMethod(self, E):
        D = np.zeros(len(E) - 2)
        for i in range(1, len(E) - 1):
            D[i - 1] = np.abs(E[i] - E[i + 1]) / np.abs(E[i - 1] - E[i])
        return np.argmin(D) + 2
    
    # K-means
    def __kmeans(self, x_set, k):
        Nu_centers = np.random.uniform(np.min(x_set), np.max(x_set), (k, 2))
        Best_Nu = np.zeros((k, 2))

        iter = 0
        while True:
            C_clusters = self.__Clustering(x_set, Nu_centers)
            Nu_centers = self.__UpdateCenters(x_set, C_clusters, Nu_centers)

            if iter == self.__iters or np.array_equal(Nu_centers, Best_Nu):
                break

            Best_Nu = np.copy(Nu_centers)
            iter += 1

        loss = self.__Error(x_set, C_clusters, Best_Nu)
        
        return loss, Best_Nu
                

    # Training model
    def fit(self, x_set):
        print('(K-means) Training...')
        if self.__elbow:
            Nu_k_centers = []
            Error_k = np.zeros(self.__size)
            
            for k in range(1, self.__size + 1):
                loss, Nu = self.__kmeans(x_set, k)

                Error_k[k - 1] = loss
                Nu_k_centers.append(Nu)
                print('(K-means) K = {}, Error - {}'.format(k, loss))

            best_k = self.__ElbowMethod(Error_k)
            self.__size = best_k
            self.__centroids = Nu_k_centers[best_k - 1]
            print('(K-means) Best k - {}'.format(best_k))
        else:
            _, Nu = self.__kmeans(x_set, self.__size)
            self.__centroids = Nu
        print('(K-means) Done!')
    
    # Predict
    def predict(self, x):
        if len(x.shape) == 1:
            return self.__Clustering(x[None, :], self.__centroids)
        else:
            return self.__Clustering(x, self.__centroids)

# Create dataset
#centers = [[-1, -1], [0, 1], [1, -1]]
#X, _ = make_blobs(n_samples=3000, centers=centers, cluster_std=0.5)

# Data Set
X = np.load('dataset.npy')

# Parametrs
MaxClusters = 10
iters = 100
isElbow = True

# Training model
model = Kmeans(K=MaxClusters, iters=iters, elbow=isElbow)
model.fit(X)
parametrs = model.get()

# Clustering
Centroids = parametrs['Centroids']
pred = model.predict(X)

# Plot
plt.figure('Clusterin')
for i in range(parametrs['K']):
    plt.plot(X[pred == i][:, 0], X[pred == i][:, 1], '.', label='С' + str(i + 1), color=np.random.randint(0, 255, 3) / 255)
plt.plot(Centroids[:, 0], Centroids[:, 1], 'k*', label='Centroids')
plt.legend()
plt.show()