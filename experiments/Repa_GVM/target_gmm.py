from experiments.target_lnpdfs.GMM import make_twoD_target, make_20D_target
from experiments.Repa_GVM.experiment_script import sample
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sample(target_dist_maker=make_20D_target, path="/tmp/iaf/planar_four_goal/lr_1r-5/Bijectors_10/layers_256_256/checkpoint",
           max_fevals=100000, samples_per_batch=100, learning_rate=5e-4, do_plots=False)