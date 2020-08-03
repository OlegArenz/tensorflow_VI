from experiments.target_lnpdfs.Planar_Robot import make_four_goal, make_single_goal
from experiments.Repa_IAF.experiment_script import sample
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    num_dimensions = 10
    # The following values are tuned to get an initial distribution similar to the configuration prior
    output_scale = [7.8,2.,2.,2.,2.,2.,2.,2.,2.,2.]
    sample(target_dist_maker=make_single_goal, path="/tmp/iaf/planar_four_goal/lr_1r-5/Bijectors_10/layers_256_256/checkpoint",
           max_fevals=int(1e7), samples_per_batch=300, learning_rate=1e-5,
           num_bijectors=15, output_scale=output_scale, hidden_layers=[128, 128], do_plots=True)

    print("done")