from experiments.target_lnpdfs.LogisticRegression import make_breast_cancer, make_german_credit
from experiments.Repa_IAF.experiment_script import sample
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset=make_breast_cancer
    sample(target_dist_maker=dataset, path="/tmp/iaf/german_credit/lr_1r-5/Bijectors_10/layers_256_256/checkpoint",
           max_fevals=100000, samples_per_batch=20, learning_rate=5e-4,
           num_bijectors=10, output_scale=40, hidden_layers=[128, 128], do_plots=True)