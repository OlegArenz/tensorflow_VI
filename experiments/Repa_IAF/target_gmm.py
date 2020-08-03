from experiments.target_lnpdfs.GMM import make_twoD_target, make_20D_target
from experiments.Repa_IAF.experiment_script import sample
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sample(target_dist_maker=make_20D_target, path="/tmp/iaf/planar_four_goal/lr_1r-5/Bijectors_10/layers_256_256/checkpoint",
           max_fevals=1000000, samples_per_batch=20, learning_rate=5e-4,
           num_bijectors=10, output_scale=4000, hidden_layers=[256,256], do_plots=True)