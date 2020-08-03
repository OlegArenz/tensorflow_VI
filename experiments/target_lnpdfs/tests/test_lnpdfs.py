from experiments.target_lnpdfs.LogisticRegression import make_breast_cancer, make_german_credit
from experiments.target_lnpdfs.Planar_Robot import make_four_goal, make_single_goal
import numpy as np

def compare_pdfs(filename, targetfun):
    dat = np.load(filename)
    samples = dat['samples']
    lnpdfs = dat['lnpdfs']
    my_lnpdfs = targetfun.log_density(samples.astype(np.float32))
    abs_errors = np.abs(my_lnpdfs - lnpdfs)
    rel_errors = abs_errors / np.abs(lnpdfs)

    ind = np.argmax(abs_errors)
    print("Maximum absolute error: {0} ({1} vs {2})".format(abs_errors[ind], my_lnpdfs[ind], lnpdfs[ind]))
    ind = np.argmax(rel_errors)
    print("Maximum relative error: {0} ({1} vs {2})".format(rel_errors[ind], my_lnpdfs[ind], lnpdfs[ind]))

    if rel_errors[ind] < 0.1:
        print("passed")
        return True
    print("Failed!")
    return False

if __name__ == "__main__":
    outcomes = []
    print("checking breast cancer...")
    outcomes.append(compare_pdfs("experiments/target_lnpdfs/tests/breast_cancer_testdat.npz", make_breast_cancer()))

    print("checking german credit...")
    outcomes.append(compare_pdfs("experiments/target_lnpdfs/tests/german_credit_testdat.npz", make_german_credit()))

    print("checking planar robot (1 goal)...")
    outcomes.append(compare_pdfs("experiments/target_lnpdfs/tests/planar1goal_testdat.npz", make_single_goal()))

    print("checking planar robot (4 goals)...")
    outcomes.append(compare_pdfs("experiments/target_lnpdfs/tests/planar4goal_testdat.npz", make_four_goal()))

    print(outcomes)