import numpy as np
from distributions.GaussianMixture import GaussianMixture
from experiments.target_lnpdfs.LogisticRegression import LNPDF
import tensorflow_probability as tfp
tfd = tfp.distributions
class GMM_LNPDF(LNPDF):
    def __init__(self, target_weights, target_means, target_covars):
        self.target_weights = target_weights
        self.target_means = target_means.astype(np.float32)
        self.target_covars = target_covars.astype(np.float32)
        self.components = [tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
                           for mu, cov in zip(self.target_means, self.target_covars)]

        self.gmm = tfd.Mixture(
                      cat=tfd.Categorical(probs=self.target_weights),
                      components=self.components)

    def log_density(self, x):
        return self.gmm.log_prob(x)

    def get_num_dimensions(self):
        return len(self.target_means[0])

    def can_sample(self):
        return True

    def sample(self, n):
        return self.gmm.sample(n)

def make_twoD_target():
    target_weights = np.array([.3, 0.3, 0.4])
    target_means = np.array([[10.0, 10.0],
                             [1.0, 1.0],
                             [5.0, -5.0]])
    #                                    c1                        c2                        c3
    target_covars = np.array([[[1., .0], [.0, 1.]],
                              [[0.1, 0.0], [0.0, 0.7]],
                              [[0.1, 0.0], [0.0, 0.7]]])
    return GMM_LNPDF(target_weights, target_means, target_covars)

def make_20D_target():
    num_true_components = 10
    num_dimensions = 20
    weights = np.ones(num_true_components) / num_true_components
    means = np.empty((num_true_components, num_dimensions))
    covs = np.empty((num_true_components, num_dimensions, num_dimensions))
    for i in range(0, num_true_components):
        means[i] = 100 * (np.random.random(num_dimensions) - 0.5)
        covs[i] = 0.1 * np.random.normal(0, num_dimensions, (num_dimensions * num_dimensions)).reshape(
            (num_dimensions, num_dimensions))
        covs[i] = covs[i].transpose().dot(covs[i])
        covs[i] += 1 * np.eye(num_dimensions)
    return GMM_LNPDF(weights, means, covs)



