#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:01:57 2019

@author: mzhong
"""

import tensorflow as tf
import numpy as np


def gaussian_log_density(samples, mean, chol_covar):
    logdet = covar_logdet(chol_covar)
    diff = samples - tf.expand_dims(mean, 0)
    t1 = tf.linalg.triangular_solve(chol_covar, tf.transpose(diff))
    exp_term = tf.reduce_sum(t1 ** 2, 0)
    return - 0.5 * (tf.cast(tf.shape(samples)[-1], exp_term.dtype) * np.log(2 * np.pi) + logdet + exp_term)


def gaussian_density(samples, mean, chol_covar):
    return tf.exp(gaussian_log_density(samples, mean, chol_covar))


def covar_logdet(chol_covar):
    return 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_covar) + 1e-20))


class Gaussian:

    def __init__(self, mean, covar, trainable=True):
        super().__init__()
        self.trainable = trainable
        self._dim = mean.shape[0]

        try:
            chol_covar = np.linalg.cholesky(covar)
        except np.linalg.LinAlgError:
            raise AssertionError("Given Covariance is not positive definite")

        if self.trainable:
            self._mean = tf.Variable(name="mean", shape=[self._dim], initial_value=np.squeeze(mean), dtype=tf.float32)
            np.fill_diagonal(chol_covar, np.log(np.diag(chol_covar)))
            self._chol_covar_raw = tf.Variable(name="chol_covar_raw", shape=[self._dim, self._dim],
                                               initial_value=chol_covar, dtype=tf.float32)
            self.trainable_variables = [self._mean, self._chol_covar_raw]
        else:
            self._mean = tf.constant(value=mean, name="mean", dtype=tf.float32)
            self._chol_covar_raw = tf.constant(value=chol_covar, name="chol_covar", dtype=tf.float32)
            self.trainable_variables = []

    @property
    def chol_covar(self):
        if self.trainable:
            return self.create_chol(self._chol_covar_raw)
        else:
            return self._chol_covar_raw

    @staticmethod
    def create_chol(chol_raw):
        x = tf.linalg.band_part(chol_raw, -1, 0)
        return tf.linalg.set_diag(x, tf.exp(tf.linalg.diag_part(x)))

    @property
    def mean(self):
        return self._mean

    @property
    def covar(self):
        return tf.matmul(self.chol_covar, self.chol_covar, transpose_b=True)

    def log_density(self, samples):
        return gaussian_log_density(samples, self._mean, self.chol_covar)

    def density(self, samples):
        return gaussian_density(samples, self._mean, self.chol_covar)

    def sample(self, num_samples):
        eps = tf.random.normal([num_samples, self._dim])
        return tf.expand_dims(self._mean, 0) + tf.matmul(eps, self.chol_covar, transpose_b=True)

    @property
    def entropy(self):
        return 0.5 * (self._dim * np.log(2 * np.e * np.pi) + covar_logdet(self.chol_covar))

    def kl(self, other_mean, other_chol_covar):
        kl = covar_logdet(other_chol_covar) - covar_logdet(self.chol_covar) - self._dim
        kl += tf.linalg.trace(tf.linalg.cholesky_solve(other_chol_covar, self.covar))
        diff = tf.expand_dims(other_mean - self._mean, -1)
        kl += tf.squeeze(tf.matmul(diff, tf.linalg.cholesky_solve(other_chol_covar, diff), transpose_a=True))
        return 0.5 * kl