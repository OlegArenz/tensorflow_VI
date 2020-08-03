import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as k
from attrdict import AttrDict
import matplotlib.pyplot as plt
plt.ion()

tfd = tfp.distributions

class BBVI:
    @staticmethod
    def get_default_config():
        return AttrDict(
            nos_of_samples=20,
            learning_rate=1e-3,
        )

    def __init__(self, config, target_log_density, model):
        self.c = config
        self._target_log_density = target_log_density
        self._model = model
        self._optimizer = k.optimizers.Adam(self.c.learning_rate)
        self.nosOfSamples = self.c.nos_of_samples
        self.LOSS = []


    def train(self, iterations):
        for i in range(iterations):
            self.LOSS.append(self._train_step().numpy())

    @tf.function
    def _train_step(self, control_variate=True):
        sample = self._model.sample(self.nosOfSamples)

        with tf.GradientTape() as tape:
            log_q_x = self._model.log_prob(sample)
            log_ratio = tf.stop_gradient(self._model.log_prob(sample) - self._target_log_density(sample))

            if control_variate:
                loss = log_q_x * tf.stop_gradient(log_ratio - tf.reduce_mean(log_ratio))
            else:
                loss = log_q_x * log_ratio

            aveloss = tf.reduce_mean(loss)
            gradients = tape.gradient(aveloss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return aveloss


