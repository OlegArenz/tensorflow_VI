from experiments.target_lnpdfs.LogisticRegression import make_breast_cancer, make_german_credit
from experiments.Repa_GVM.experiment_script import sample
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset=make_german_credit
    sample(target_dist_maker=dataset, path="/tmp/iaf/german_credit/lr_1r-5/Bijectors_10/layers_256_256/checkpoint",
           max_fevals=1000000, samples_per_batch=500, learning_rate=1e-2, scale_diag=10., do_plots=True)