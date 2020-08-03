from distributions.GaussianMixture import GaussianMixture as GMM
from variational_inference.bbvi.BBVI import BBVI
import numpy as np
from time import time as time
import os
import tensorflow_probability as tfp
tfd = tfp.distributions

def construct_initial_mixture(num_dimensions, num_initial_components, prior_scale):
    if np.isscalar(prior_scale):
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(num_dimensions), scale_identity_multiplier=prior_scale)
    else:
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(num_dimensions), scale_diag=prior_scale)

    initial_covs = prior.covariance().numpy().astype(np.float32) # use the same initial covariance that was used for sampling the means

    weights = np.ones(num_initial_components) / num_initial_components
    means = np.empty((num_initial_components, num_dimensions), dtype=np.float32)
    covs = np.empty((num_initial_components, num_dimensions, num_dimensions), dtype=np.float32)
    for i in range(0, num_initial_components):
        if num_initial_components == 1:
            means[i] = np.zeros(num_dimensions)
        else:
            means[i] = prior.sample(1).numpy()
        covs[i] = initial_covs

    gmm = GMM(weights, means, covs, trainable=True)
    return gmm


def sample(target_dist_maker, model,
           path, max_fevals, samples_per_batch, learning_rate, do_plots=False):
    if do_plots:
        import matplotlib.pyplot as plt
        plt.ion()

    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    target_distribution = target_dist_maker()

    config = BBVI.get_default_config()
    config['nos_of_samples'] = samples_per_batch
    config['learning_rate'] = learning_rate

    bbvi = BBVI(config, target_distribution.log_density, model)
    bbvi.target_dist = target_distribution

    timestamps = [0]
    samples = [model.sample(2000).numpy()]
    nfevals = [0]
    model_lnpdfs = [model.log_density(samples[-1])]
    i=0
    start = time()

    # aim for roughly 200 checkpoints / data points per plot
    iterations_per_checkpoint = int(np.ceil(max_fevals / (200 * samples_per_batch)))
    while True:
        bbvi.train(iterations_per_checkpoint)
        timestamps.append(time() - start)
        samples.append(model.sample(2000).numpy())
        nfevals.append((i+1) * iterations_per_checkpoint * samples_per_batch)
        model_lnpdfs.append(model.log_density(samples[-1]))
        mean_reward = np.mean(target_distribution.log_density(samples[-1]))
        print("Checkpoint {:3d} | FEVALS: {:10d} | loss: {:05.05f} | avg. sample logpdf: {:05.05f} | ELBO: {:05.05f}".format(
            i, nfevals[-1], bbvi.LOSS[-1], mean_reward, mean_reward - np.mean(model_lnpdfs[-1])))
        if hasattr(target_distribution, "target_weights"):
            np.savez(path + "_processed_data.npz", samples=samples,  model_lnpdfs=model_lnpdfs, timestamps=timestamps,
                     true_weights=target_distribution.target_weights, true_means=target_distribution.target_means,
                     true_covs=target_distribution.target_covars, fevals=nfevals)
        else:
            np.savez(path+"_processed_data.npz", model_lnpdfs=model_lnpdfs, samples=samples, timestamps=timestamps, fevals=nfevals)
        if nfevals[-1] >= max_fevals:
            return
        if do_plots:
            # weights = bbvi._model.mixture_dist.probabilities.numpy()
            # means = np.array([comp.mean.numpy() for comp in bbvi._model.components])
            # print("weights: " + str(weights))
            # print("means: " + str(means))

            plt.figure(1)
            plt.clf()
            if target_distribution.can_sample():
                target_samps = target_distribution.sample(2000)
                plt.plot(target_samps[:,0], target_samps[:,1], 'x')

                plt.plot(samples[-1][:,0], samples[-1][:,1], 'x')
            else:
                from plotting.visualize_n_link import visualize_mixture
                visualize_mixture(np.ones(100), samples[-1][:100])
            plt.pause(0.001)
        i+=1

    print("done")

