from tensorflow import keras as k
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors

from attrdict import AttrDict

''' Class for minimizing the KL to a target distribution using the reparameterization trick '''
class REPA_KL_Minimizer:
    @staticmethod
    def get_default_config():
        return AttrDict(
            nos_of_samples=200,
            learning_rate=1e-5,
        )

    def __init__(self, config, target_log_density, model):
        self.c = config

        self._target_log_density = target_log_density
        self._model = model
        _ = self._model.sample(3)

        self._model.sample(1)
        self.trainable_variables = self._model.trainable_variables
        self._optimizer = k.optimizers.Adam(self.c.learning_rate)

        self.nosOfSamples = self.c.nos_of_samples
        self.LOSS = []


    def train(self, iters):
        for i in range(iters):
            self.LOSS.append(self._train_step().numpy())

    """ Minimize KL to target distribution using reparameterization trick """
    @tf.function
    def _train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            sample = self._model.sample(self.nosOfSamples)
            loss = (self._model.log_density(sample) - self._target_log_density(sample))
            aveloss = tf.reduce_mean(loss)
            gradients = tape.gradient(aveloss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return aveloss


