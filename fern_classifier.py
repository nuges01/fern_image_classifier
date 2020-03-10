"""
Simple fern ensemble classifier for images. (c) Olusegun Oshin.
Adapted from “Fast Keypoint Recognition in Ten Lines of Code.” by Ozuysal M, Fua P & Lepetit V.
"""
import numpy as np
from scipy.stats import entropy


class Node:
    """
    A node is a (simple) test on data inputs. This node implementation has been extended
    to quantize test results with additional symbols.
    """

    def __init__(self, data_dims, base=2, test_threshold=0.25):
        if not len(data_dims) == 2:
            raise Exception('Data dimensions must be 2.')
        if base < 2 or base > 4:
            raise ValueError('Nodes support only binary, ternary and quaternary tests.')
        if test_threshold < 0 or test_threshold >= 1:
            raise ValueError('Node test threshold must be in the range [0,1)')
        self.base = base
        self.test_points = (np.random.randint(0, data_dims[0]), np.random.randint(0, data_dims[1]),
                            np.random.randint(0, data_dims[0]), np.random.randint(0, data_dims[1]))
        self.threshold = test_threshold

    def perform_test(self, data):
        # Performs the node test. Can return binary, ternary or quaternary results
        if not data.dtype == np.uint8 or not len(data.shape) == 2:
            raise Exception('Node tests can only be performed on images of channel=1')

        val = (int(data[self.test_points[0], self.test_points[1]]) -
               int(data[self.test_points[2], self.test_points[3]])) / 255
        if self.base == 2:  # Binary test
            return str(int(val > 0))
        elif self.base == 3:  # Ternary test
            if val > self.threshold:
                return '2'
            elif val <= -self.threshold:
                return '0'
            else:
                return '1'
        elif self.base == 4:  # Quaternary test
            if val > self.threshold:
                return '3'
            elif 0 < val <= self.threshold:
                return '2'
            elif val <= -self.threshold:
                return '0'
            else:
                return '1'
        else:
            raise NotImplementedError('Nodes support only binary, ternary and quaternary tests.')


def jsd(p, q, base=np.e):
    # Calculates Jensen-Shannon Divergence
    m = 1. / 2 * (p + q)
    return entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.


class Fern:
    """
    A Fern is a collection of nodes. Node results are combined to update class distributions
    or retrieve class posterior probabilities. This implementation is extended to obtain
    a Jensen-Shannon Divergence based score on class distribution and calculate error estimates.
    """

    def __init__(self, num_nodes, dims, num_classes, map_label_to_class, base):
        self.num_nodes = num_nodes
        self.dims = dims
        self.num_classes = num_classes
        self.map_label_to_class = map_label_to_class
        self.base = base
        self.class_distributions = np.ones((self.num_classes, self.base ** self.num_nodes))  # 1 => Dirichlet prior
        self.class_dist_normed = None
        self.fern_score = None
        self.error_estimate = None
        self.nodes = []
        for _ in range(self.num_nodes):
            self.nodes.append(Node(self.dims, self.base))

    def __get_node_results(self, data):
        # Performs tests and combine results of all nodes to get decimal symbol
        nary_results = []
        for node in self.nodes:
            nary_results.append(node.perform_test(data))
        return int(''.join(nary_results), self.base)

    def update_distribution(self, data, label):
        # Updates distribution for the class
        decimal_val = self.__get_node_results(data)
        label_idx = self.map_label_to_class[label]
        self.class_distributions[label_idx, decimal_val] += 1

    def normalize_distributions(self):
        # Calculate posterior probabilities
        row_sums = self.class_distributions.sum(axis=1)
        self.class_dist_normed = self.class_distributions / row_sums[:, np.newaxis]

    def get_class_probs(self, data):
        # Get all class probabilities based on node results
        dist_bin = self.__get_node_results(data)
        return self.class_dist_normed[:, dist_bin]

    def calc_fern_score(self):
        # Estimates scores for fern using pairwise Jensen-Shannon Divergence on class distributions
        self.fern_score = 0
        for cls_a in range(self.num_classes):
            pos_cls = self.class_dist_normed[cls_a, :]
            for cls_b in range(cls_a):
                neg_cls = self.class_dist_normed[cls_b, :]
                self.fern_score += jsd(pos_cls, neg_cls)
        return self.fern_score

    def calc_error_estimate(self, X, y):
        # Estimates classification error for a fern.
        # Should be used with OOB set when using bootstrap aggregation
        y_predicted = []
        for idx in range(X.shape[0]):
            x = X[idx, :]
            y_predicted.append(np.argmax(self.get_class_probs(x)))
        assert len(y_predicted) == y.shape[0]
        error = [a for (a, b) in zip(y_predicted, y) if a != b]
        self.error_estimate = len(error) / len(y_predicted)
        return self.error_estimate


class FernEnsembleClassifier:
    """
    Random Ferns are an ensemble classifier. FernEnsembleClassifier uses standard SKLearn fit/predict format
    """
    def __init__(self, num_ferns, num_nodes, data_dims, base=2, use_bagging=False, get_error_estimates=False):
        if get_error_estimates and not use_bagging:
            raise ValueError('Error estimates can only be obtained when use_bagging=True.')
        self.num_ferns = num_ferns
        self.num_nodes = num_nodes
        self.data_dims = data_dims
        self.base = base
        self.use_bagging = use_bagging
        self.get_error_estimates = get_error_estimates
        self.ferns = []

    def fit(self, X, y):
        if not X.shape[0] == y.shape[0]:
            raise Exception('Number of labels must match number of data points')

        labels = list(set(y))
        num_classes = len(labels)
        # map_label_to_class (and vice versa) allows non-contiguous, non-zero indexed class labels in input data
        map_label_to_class = {k: v for k, v in zip(labels, list(range(num_classes)))}
        self.map_class_to_label = dict((reversed(item) for item in map_label_to_class.items()))
        fern_count = 0
        while fern_count < self.num_ferns:
            # Create Fern object, update with training data and normalize distribution
            fern = Fern(self.num_nodes, self.data_dims, num_classes, map_label_to_class, self.base)
            if self.use_bagging:
                # Bootstrap aggregating: sample data with replacement
                sample_idx = np.random.choice(X.shape[0], X.shape[0])
                X = X[sample_idx, :]
                y = y[sample_idx]
            for x, cls in zip(X, y):
                fern.update_distribution(x, cls)
            fern.normalize_distributions()

            if self.use_bagging and self.get_error_estimates:
                # When using bagging, use out-of-bag (OOB) set to calculate error estimates
                oob = np.where(np.isin(list(range(X.shape[0])), sample_idx) == False)[0]
                fern.calc_error_estimate(X[oob, :], y[oob])
            fern_count += 1
            self.ferns.append(fern)
            print('Trained {}/{}.'.format(fern_count, self.num_ferns))

    def predict(self, X):
        if not X[0].shape == self.data_dims:
            raise Exception('Test and training data dimensions do not match.')

        y = []
        for idx in range(X.shape[0]):
            fern_posterior_resp = []
            for fern in self.ferns:
                fern_posterior_resp.append(fern.get_class_probs(X[idx, :]))  # Get class posteriors for each fern
            ferns_resp = np.stack(fern_posterior_resp, axis=1)
            per_class_posterior = np.prod(ferns_resp, axis=1)                # Multiply posteriors across ferns
            max_class_idx = np.argmax(per_class_posterior)                   # Get class with highest value
            y.append(self.map_class_to_label[max_class_idx])
        return np.array(y)


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix, classification_report

    from keras.datasets.mnist import load_data
    (X_train, y_train), (X_test, y_test) = load_data()

    classifier = FernEnsembleClassifier(100, 9, X_train[0].shape, base=3)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))
