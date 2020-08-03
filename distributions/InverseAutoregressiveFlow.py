import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

class IAF(tf.keras.models.Model):
    def __init__(self, *, output_dim, num_bijectors, hidden_layers, output_scale=100, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim

        self.prior_mean = tf.Variable(0 * tf.ones([self.output_dim]), name="prior_mean")
        self.prior_scale = tf.Variable(.1 * tf.ones([self.output_dim]), name="prior_mean")
        self.prior = tfd.MultivariateNormalDiag(loc=self.prior_mean,
                                               scale_diag=self.prior_scale, allow_nan_stats=False)

        # Define the bijectors (IAF modules)
        num_bijectors = num_bijectors
        self.bijectors=[]
        self.trainable_nets = []
        for i in range(num_bijectors):
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_layers)
            self.trainable_nets.append(made)
            self.bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(made)))
            self.bijectors.append(tfb.Permute(permutation=np.arange(output_dim-1, -1, -1)))

        if np.isscalar(output_scale):
            self.bijectors.append(tfp.bijectors.Scale(scale=output_scale * tf.ones([self.output_dim])))
        else:
            self.bijectors.append(tfp.bijectors.Scale(scale=tf.convert_to_tensor(output_scale)))
        self.bijector = tfb.Chain(list(reversed(self.bijectors)))

        # construct the normalizing flow
        self.flow = tfd.TransformedDistribution(distribution=self.prior, bijector=self.bijector)
        self.call(np.zeros([1, output_dim])) # for actually building the model


    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    def log_density(self, samples):
        return tf.transpose(self.flow.log_prob(samples))

    def density(self, samples):
        return tf.math.exp(self.log_density(samples))

    def sample(self, num):
        return self.flow.sample(num)

