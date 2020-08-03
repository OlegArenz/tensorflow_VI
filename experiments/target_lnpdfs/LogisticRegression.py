import tensorflow as tf
import numpy as np
from experiments.target_lnpdfs.Lnpdf import LNPDF
from math import pi

class LogisticRegression(LNPDF):
    def __init__(self, dataset_id):
        if dataset_id == "breast_cancer":
            data = np.loadtxt("experiments/target_lnpdfs/datasets/breast_cancer.data")
            X = data[:, 2:]
            X /= np.std(X, 0)[np.newaxis, :]
            X = np.hstack((np.ones((len(X), 1)), X))
            self.data = tf.cast(X, tf.float32)
            self.labels = data[:,1]
            self.num_dimensions = self.data.shape[1]
            self.prior_std = tf.constant(10.)
            self.prior_mean = tf.constant(0.)
        elif dataset_id == "german_credit":
            data = np.loadtxt("experiments/target_lnpdfs/datasets/german.data-numeric")
            X = data[:, :-1]
            X /= np.std(X, 0)[np.newaxis, :]
            X = np.hstack((np.ones((len(X), 1)), X))
            self.data = tf.cast(X, tf.float32)
            self.labels = data[:, -1] - 1
            self.num_dimensions = self.data.shape[1]
            self.prior_std = tf.constant(10.)
            self.prior_mean = tf.constant(0.)

    def get_num_dimensions(self):
        return self.num_dimensions

    def log_density(self, x):
        features = -tf.matmul(self.data, tf.transpose(x))
        log_likelihoods = tf.reduce_sum(tf.where(self.labels==1, tf.transpose(tf.math.log_sigmoid(features)), tf.transpose(tf.math.log_sigmoid(features) - features)), axis=1)
        log_prior = tf.reduce_sum(-tf.math.log(self.prior_std) - 0.5 * tf.math.log(2. * pi) - 0.5 * tf.math.square((x - self.prior_mean) / self.prior_std), axis=1)
        log_posterior = log_likelihoods + log_prior
        return log_posterior

def make_breast_cancer():
    return LogisticRegression("breast_cancer")

def make_german_credit():
    return LogisticRegression("german_credit")