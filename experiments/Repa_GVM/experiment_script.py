from distributions.Gaussian import Gaussian
from variational_inference.Repa_KL_Minimization.Repa_KL_Minimization import REPA_KL_Minimizer

import numpy as np
from time import time as time
import os
from datetime import datetime

def sample(target_dist_maker, path, max_fevals, samples_per_batch, learning_rate, scale_diag, do_plots=False):
    if do_plots:
        import matplotlib.pyplot as plt
        plt.ion()
    start_time = datetime.now().strftime('%Y-%m-%d%_H:%M:%S.%f')
    path = path+"/"+str(start_time)

    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    target_distribution = target_dist_maker()

    config = REPA_KL_Minimizer.get_default_config()
    config['nos_of_samples'] = samples_per_batch
    config['learning_rate'] = learning_rate

    output_dim=target_distribution.get_num_dimensions()
    mean = np.zeros(output_dim)
    covar = np.square(scale_diag) * np.eye(output_dim)

    model = Gaussian(mean, covar, trainable=True)

    repa_GVM = REPA_KL_Minimizer(config,
                       target_distribution.log_density,
                       model)
    repa_GVM.target_dist = target_distribution
    model_lnpdfs = []
    timestamps = []
    samples = []
    nfevals = []
    i=0

    start = time()
    # aim for roughly 200 checkpoints / data points per plot
    iterations_per_checkpoint = int(np.ceil(max_fevals / (200 * samples_per_batch)))
    while True:
        repa_GVM.train(iterations_per_checkpoint)
        samps = model.sample(2000).numpy()
        if do_plots:
            plt.figure(1)
            plt.clf()
            from plotting.visualize_n_link import visualize_mixture
            visualize_mixture(np.ones(100), samps[:100])
            plt.pause(0.001)
        timestamps.append(time() - start)
        samples.append(samps)
        nfevals.append((i+1) * iterations_per_checkpoint * samples_per_batch)
        model_lnpdfs.append(model.log_density(samples[-1]))
        print("Checkpoint {:3d} | FEVALS: {:10d} | KL_loss: {:05.05f} | avg. sample logpdf: {:05.05f}".format(
            i, nfevals[-1], repa_GVM.LOSS[-1], np.mean(target_distribution.log_density(samples[-1]))))
        if hasattr(target_distribution, "target_weights"):
            np.savez(path + "_processed_data.npz", samples=samples, model_lnpdfs=model_lnpdfs, timestamps=timestamps,
                     true_weights=target_distribution.target_weights, true_means=target_distribution.target_means,
                     true_covs=target_distribution.target_covars, fevals=nfevals)
        else:
            np.savez(path+"_processed_data.npz", samples=samples, model_lnpdfs=model_lnpdfs, timestamps=timestamps, fevals=nfevals)
        if nfevals[-1] >= max_fevals:
            return
        i+=1
        if np.any(np.isnan(samps)):
            break

    print("done")