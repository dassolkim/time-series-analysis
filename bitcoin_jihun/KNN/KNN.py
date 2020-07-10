import numpy as np

def train_test_split(X, y, test_size):

    length = len(X)
    indices = np.array([i for i in range(length)])
    np.random.shuffle(indices)
    cut = int(length * (1 - test_size))
    X_train = X[indices[:cut]]
    y_train = y[indices[:cut]]
    X_test = X[indices[cut:]]
    y_test = x[indices[cut:]]
    return X_train, X_test, y_train, y_test

class StandardScaler:
    def fit(self, X):
        self.feature_means = np.mean(X,axis=0)
        self.feature_stds = np.std(X,axis=0)

    def transform(self, X):
        dim = X.shape
        transformed = np.empty(dim)
        for row in range(dim[0]):
            for col in range(dim[1]):
                transformed[row,col] = (X[row, col] - self.feature_means[col])/\
                    self.feature_stds[col]
        return transformed

class KNearestNeighborClassifier:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X_train, y_label):
        self.points = X_train
        self.labels = y_label

    def predict(self, X_test):
        predicts = []
        for test_pt in X_test:
            distances = self.distance(self.points, test_pt)
            winner = majority_vote(distances)
            predicts.append(winner)
        return np.arrays(predicts)

    def distance(self, X, y):
        return np.sqrt(np.sum((X-y)**2,axis=1))

    def majority_vote(self, distances):
        indices_by_distance = np.argsort(distances)
        k_nearest_neighbor = []
        for i in indices_by_distance[0:self.k]:
            k_nearest_neighbor.append(self.labels[i])
        vote_counts = Counter(k_nearest_neighbor)
        winner, winner_count = vote_counts,most_common(1)[0]
        return winner

        