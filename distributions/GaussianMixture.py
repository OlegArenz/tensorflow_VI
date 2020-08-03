import tensorflow as tf
import os
from distributions.Gaussian import Gaussian
from distributions.Categorical import Categorical
import numpy as np

class GaussianMixture:
    def __init__(self, weights, means, covars, trainable=True):
        self._num_components, self._x_dim = means.shape

        self._mixture_dist = Categorical(weights, trainable=trainable)

        self._components = \
            [Gaussian(means[i], covars[i], trainable=trainable) for i in range(self._num_components)]

        self.trainable_variables = self._mixture_dist.trainable_variables.copy()
        for c in self._components:
            self.trainable_variables += c.trainable_variables

    @property
    def components(self):
        return self._components

    def density(self, x):
        return tf.exp(self.log_density(x))

    def log_prob(self, x):
        return tf.reduce_logsumexp(tf.math.log(self.mixture_dist.probabilities)[:, None]
                                   + tf.stack([comp.log_density(x) for comp in self.components]), axis=0)

    def log_density(self, x):
        return self.log_prob(x)

    def sample(self, num_samples):
        """non-reparameterizable, exact sampling"""
        modes = self._mixture_dist.sample(num_samples)
        samples = []
        for i, c in enumerate(self._components):
            samples.append(c.sample(tf.math.count_nonzero(tf.equal(modes, i))))
        return tf.random.shuffle(tf.concat(samples, 0))

    def sample_gumble_softmax(self, num_samples, gs_temperature):
        """reparameterizable sampling (mixture is sampled approximately using gumbel softmax"""
        one_hot_modes = self._mixture_dist.sample_gumble_softmax(num_samples, gs_temperature=gs_temperature, hard=True)
        c_samples = tf.concat([tf.expand_dims(c.sample(num_samples), 1) for c in self.components], 1)
        return tf.transpose(tf.reduce_sum(tf.expand_dims(one_hot_modes, -1) * c_samples, 1))

    @property
    def num_components(self):
        return self._num_components

    @property
    def mixture_dist(self):
        return self._mixture_dist

    @property
    def weights(self):
        return self._mixture_dist.probabilities

    @property
    def means(self):
        return tf.stack([c.mean for c in self._components], axis=0)

    @property
    def covars(self):
        return tf.stack([c.covar for c in self._components], axis=0)

    @property
    def chol_covars(self):
        return tf.stack([c.chol_covar for c in self._components], axis=0)

    @property
    def parameters(self):
        return self.weights, self.means, self.chol_covars

    def save(self, path, filename):
        model_dict = {"weights": np.array(self.weights),
                      "means": np.array(self.means),
                      "covars": np.array(self.covars)}
        np.savez_compressed(os.path.join(path, filename + ".npz"), **model_dict)

    @staticmethod
    def load_gmm(path, filename, trainable):
        model_path = os.path.join(path, filename + ".npz")
        model_dict = dict(np.load(model_path))
        return GaussianMixture(model_dict["weights"], model_dict["means"], model_dict["covars"], trainable=trainable)
