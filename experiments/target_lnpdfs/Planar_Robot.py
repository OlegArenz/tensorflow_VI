import numpy as np
from experiments.target_lnpdfs.Lnpdf import LNPDF
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

class PlanarRobot(LNPDF):
    def __init__(self, num_links, num_goals, prior_std=2e-1, likelihood_std=1e-2):
        self._num_dimensions = num_links
        prior_stds = prior_std * np.ones(num_links)
        prior_stds[0] = 1.
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(num_links),  scale_diag=prior_stds.astype(np.float32))
        self.link_lengths = np.ones(self._num_dimensions)

        if num_goals == 1:
            self.goal_Gaussian = tfd.MultivariateNormalDiag(loc=[0.7 * self._num_dimensions, 0],
                                                            scale_identity_multiplier=likelihood_std)
            self.likelihood = self.goal_Gaussian.log_prob
        elif num_goals == 4:
            self.goal_Gaussian1 = tfd.MultivariateNormalDiag(loc=[0.7 * self._num_dimensions, 0],
                                                            scale_identity_multiplier=likelihood_std)
            self.goal_Gaussian2 = tfd.MultivariateNormalDiag(loc=[-0.7 * self._num_dimensions, 0],
                                                            scale_identity_multiplier=likelihood_std)
            self.goal_Gaussian3 = tfd.MultivariateNormalDiag(loc=[0, 0.7 * self._num_dimensions],
                                                            scale_identity_multiplier=likelihood_std)
            self.goal_Gaussian4 = tfd.MultivariateNormalDiag(loc=[0, -0.7 * self._num_dimensions],
                                                            scale_identity_multiplier=likelihood_std)
            self.likelihood = lambda pos: tf.reduce_max(tf.stack(
                (self.goal_Gaussian1.log_prob(pos),
                 self.goal_Gaussian2.log_prob(pos),
                 self.goal_Gaussian3.log_prob(pos),
                 self.goal_Gaussian4.log_prob(pos))), axis=0)
        else:
            raise ValueError

    def get_num_dimensions(self):
        return self._num_dimensions

    def log_density(self, theta):
        y = tf.zeros(len(theta))
        x = tf.zeros(len(theta))
        for i in range(0, self._num_dimensions):
            y += self.link_lengths[i] * tf.math.sin(tf.reduce_sum(theta[:, :i+1], axis=1))
            x += self.link_lengths[i] * tf.math.cos(tf.reduce_sum(theta[:, :i+1], axis=1))
        return self.prior.log_prob(theta) + self.likelihood(tf.stack((x,y), axis=1))

def make_single_goal():
    return PlanarRobot(10, 1)

def make_four_goal():
    return PlanarRobot(10, 4)