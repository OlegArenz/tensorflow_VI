from experiments.target_lnpdfs.Planar_Robot import make_four_goal, make_single_goal
from experiments.Repa_GVM.experiment_script import sample
import os
import numpy as np

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    num_dimensions = 10
    conf_likelihood_var = 4e-2 * np.ones(num_dimensions)
    conf_likelihood_var[0] = 1

    sample(target_dist_maker=make_single_goal, path="/tmp/repa_gvm/planar_1/samps30/lr_2e-3/6",
           max_fevals=int(1e7), samples_per_batch=30, learning_rate=2e-3, scale_diag=np.sqrt(conf_likelihood_var), do_plots=True)

    print("done")