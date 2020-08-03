from experiments.target_lnpdfs.LogisticRegression import make_breast_cancer, make_german_credit
from experiments.BBVI.experiment_script import sample, construct_initial_mixture
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    breast_cancer = True
    if breast_cancer:
        dataset=make_breast_cancer
        model = construct_initial_mixture(31, 1, 10)
        path="/tmp/bbvi/breast_cancer/"
    else: # German credit
        dataset=make_german_credit
        model = construct_initial_mixture(25, 1, 10)
        path="/tmp/bbvi/german_credit/"

    sample(target_dist_maker=dataset, path=path+"lr_1r-5/samps_per_batch_20/checkpoint",
           model=model, max_fevals=10000000, samples_per_batch=30, learning_rate=1e-3, do_plots=True)