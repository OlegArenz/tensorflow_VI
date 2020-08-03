from experiments.target_lnpdfs.Planar_Robot import make_four_goal, make_single_goal
from experiments.BBVI.experiment_script import sample, construct_initial_mixture
import os
import numpy as np

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    num_dimensions = 10
    conf_likelihood_var = 4e-2 * np.ones(num_dimensions)
    conf_likelihood_var[0] = 1

    four_goal = False
    if four_goal:
        target_fn = make_four_goal
        model = construct_initial_mixture(num_dimensions=num_dimensions, num_initial_components=10, prior_scale=np.sqrt(conf_likelihood_var))
    else:
        target_fn = make_single_goal
        model = construct_initial_mixture(num_dimensions=num_dimensions, num_initial_components=1, prior_scale=np.sqrt(conf_likelihood_var))

    sample(target_dist_maker=target_fn, path="/tmp/bbvi_gvm/planar_1/samps1000/lr_8e-4/6",
           model=model, max_fevals=int(100000000), samples_per_batch=1000, learning_rate=8e-4, do_plots=True)

    print("done")
