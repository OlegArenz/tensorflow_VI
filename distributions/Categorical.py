import tensorflow as tf

class Categorical:
    def __init__(self, initial_logits, trainable=True):
        self._dim = initial_logits.shape[0]
        self.trainable = trainable
        
        if self.trainable:
            self._logits = tf.Variable(name="weights", shape=[self._dim], initial_value=initial_logits, dtype=tf.float32)
            self._trainable_variables = [self._logits]
        else:
            self._logits = tf.constant(value=initial_logits, name="weights", dtype=tf.float32)
            self._trainable_variables = []
            
    @property
    def probabilities(self):
        return tf.nn.softmax(self.logits)
    
    @property
    def logits(self):
        return self._logits
    
    @property
    def dim(self):
        return self.logits.shape[0]
    
    @logits.setter
    def logits(self,_logits):
        if self.trainable:
            self._logits = tf.Variable(name="weights", shape=[_logits.shape[0]], initial_value=_logits, dtype=tf.float32)
        else:
            self._logits = tf.constant(name="weights", value=_logits, dtype=tf.float32)
    
    @property
    def trainable_variables(self):
        if self.trainable:
            self._trainable_variables = [self.logits]
        else:
            self._trainable_variables = []
        return self._trainable_variables

    def sample(self, num_samples):
        """Non reparametrizable exact sampling """
        thresholds = tf.expand_dims(tf.cumsum(self.probabilities), 0)
        n = tf.random.uniform(shape=[num_samples, 1], minval=0.0, maxval=1.0)
        idx = tf.where(tf.less(n, thresholds), tf.range(self.dim) * tf.ones(thresholds.shape, dtype=tf.int32),
                       self.dim * tf.ones(thresholds.shape, dtype=tf.int32))
        return tf.reduce_min(idx, -1)

    def entropy(self):
        return - tf.reduce_sum(self.probabilities * tf.math.log(self.probabilities + 1e-25))

    def kl(self, other_probabilities):
        log_prob_ratio = tf.math.log(self.probabilities + 1e-25) - tf.math.log(other_probabilities + 1e-25)
        return tf.reduce_sum(self.probabilities * log_prob_ratio)
