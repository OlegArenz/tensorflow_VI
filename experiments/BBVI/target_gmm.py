from experiments.target_lnpdfs.GMM import make_twoD_target, make_20D_target
from experiments.BBVI.experiment_script import sample, construct_initial_mixture
import numpy as np
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    twoD = False
    if twoD:
        # A simple 2D problem for visualization
        target_fn = make_twoD_target
        model = construct_initial_mixture(num_dimensions=2, num_initial_components=10, prior_scale=10)
    else:
        # Experiment from paper
        target_fn = make_20D_target
        model = construct_initial_mixture(num_dimensions=20, num_initial_components=100, prior_scale=np.sqrt(1000))

    sample(target_dist_maker=target_fn,  path="/tmp/iaf/gmm/lr_1e-3/samps_per_batch_20/checkpoint", model=model,
           max_fevals=20000, samples_per_batch=1000, learning_rate=5e-3, do_plots=True)

    print("done")