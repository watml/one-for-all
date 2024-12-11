import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
from utils.load_datasets import load_dataset
from utils.utilityFuncs import gameTraining
from utils.estimators import runEstimator
from utils.funcs import os_lock
import traceback
import numpy as np
import argparse

###############
root = os.path.join("exp", "compare_estimators;nue_avg=20000")
dir_format = "metric={};semivalue={}_{};estimator={}"
n_process = 100
nue_per_proc = 500
nue_avg = 20000
nue_track_avg = 200

seeds = np.arange(30)
datasets = ["FMNIST", "MNIST", "wind", "iris"]
metrics = ["cross_entropy"]
# datasets = ["FMNIST"]
# metrics = ["cross_entropy"]
n_valued = 24
n_perf = 24
semivalues = {
    ("shapley", None) : ["GELS", "GELS_paired", "GELS_shapley", "ARM", "sampling_lift", "kernelSHAP", "kernelSHAP_paired",
                         "unbiased_kernelSHAP", "GELS_shapley_paired", "permutation", "group_testing", "complement",
                         "sampling_lift_paired", "simSHAP", "OFA", "OFA_fixed", "OFA_fixed_paired", "OFA_baseline",
                         "permutation_paired", "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                         "ARM_banzhaf", "WGELS_banzhaf", "WGELS_banzhaf_paired", "OFA_baseline_paired", "improved_AME",
                         "modified_leverage", "modified_leverage_paired", "test_leverage_paired"],

    ("beta_shapley", (2, 2)) : ["ARM", "sampling_lift", "WSL", "AME", "GELS", "GELS_ranking", "sampling_lift_paired",
                                "WSL_paired", "AME_paired", "GELS_paired", "GELS_ranking_paired", "OFA", "OFA_fixed",
                                "OFA_fixed_paired", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR",
                                "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation",
                                "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley", "WGELS_shapley",
                                "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired", "OFA_baseline_paired",
                                "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("beta_shapley", (4, 1)): ["ARM", "sampling_lift", "WSL", "GELS", "GELS_ranking", "WSL_paired", "OFA", "OFA_fixed",
                               "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR", "weighted_MSR_paired",
                               "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation", "weighted_permutation_paired",
                               "ARM_banzhaf", "ARM_shapley", "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf",
                               "WGELS_banzhaf_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],


    ("beta_shapley", (8, 8)) : ["ARM", "sampling_lift", "WSL", "AME", "GELS", "GELS_ranking", "sampling_lift_paired",
                                "WSL_paired", "AME_paired", "GELS_paired", "GELS_ranking_paired", "OFA", "OFA_fixed",
                                "OFA_fixed_paired", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR",
                                "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation",
                                "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley", "WGELS_shapley",
                                "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired", "OFA_baseline_paired",
                                "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("beta_shapley", (16, 1)): ["ARM", "sampling_lift", "WSL", "GELS", "GELS_ranking", "WSL_paired", "OFA", "OFA_fixed",
                                "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR", "weighted_MSR_paired",
                                "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation", "weighted_permutation_paired",
                                "ARM_banzhaf", "ARM_shapley", "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf",
                                "WGELS_banzhaf_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("beta_shapley", (1, 4)): ["ARM", "sampling_lift", "WSL", "GELS", "GELS_ranking", "WSL_paired", "OFA", "OFA_fixed",
                               "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR", "weighted_MSR_paired",
                               "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation", "weighted_permutation_paired",
                               "ARM_banzhaf", "ARM_shapley", "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf",
                               "WGELS_banzhaf_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("beta_shapley", (1, 16)): ["ARM", "sampling_lift", "WSL", "GELS", "GELS_ranking", "WSL_paired", "OFA", "OFA_fixed",
                                "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR", "weighted_MSR_paired",
                                "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation", "weighted_permutation_paired",
                                "ARM_banzhaf", "ARM_shapley", "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf",
                                "WGELS_banzhaf_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("beta_shapley", (4, 16)): ["ARM", "sampling_lift", "WSL", "GELS", "GELS_ranking", "WSL_paired", "OFA", "AME", "OFA_fixed",
                                "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR", "weighted_MSR_paired",
                                "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation", "weighted_permutation_paired",
                                "ARM_banzhaf", "ARM_shapley", "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf",
                                "WGELS_banzhaf_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("beta_shapley", (16, 4)): ["ARM", "sampling_lift", "WSL", "GELS", "GELS_ranking", "WSL_paired", "OFA", "AME", "OFA_fixed",
                                "OFA_optimal", "OFA_optimal_paired", "OFA_baseline", "weighted_MSR", "weighted_MSR_paired",
                                "WSL_banzhaf", "WSL_banzhaf_paired", "weighted_permutation", "weighted_permutation_paired",
                                "ARM_banzhaf", "ARM_shapley", "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf",
                                "WGELS_banzhaf_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.5) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "sampling_lift_paired", "WSL_paired", "AME_paired", "MSR_paired", "GELS_paired",
                                 "GELS_ranking_paired", "OFA_fixed", "OFA_fixed_paired", "OFA_optimal", "OFA_optimal_paired",
                                 "OFA_baseline", "weighted_permutation", "weighted_permutation_paired", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired",
                                 "improved_AME"],

    ("weighted_banzhaf", 0.2) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.4) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.6) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.8) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.1) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.3) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.7) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],

    ("weighted_banzhaf", 0.9) : ["ARM", "sampling_lift", "WSL", "AME", "MSR", "GELS", "GELS_ranking", "OFA",
                                 "WSL_paired", "OFA_fixed", "OFA_optimal", "OFA_optimal_paired", "OFA_baseline",
                                 "weighted_MSR", "weighted_MSR_paired", "WSL_banzhaf", "WSL_banzhaf_paired",
                                 "weighted_permutation", "weighted_permutation_paired", "ARM_banzhaf", "ARM_shapley",
                                 "WGELS_shapley", "WGELS_shapley_paired", "WGELS_banzhaf", "WGELS_banzhaf_paired",
                                 "OFA_baseline_paired", "SHAP_IQ", "SHAP_IQ_paired", "improved_AME"],
}
###############


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", action="store_true",
                                     help="to compute the exact values if specified, and run estimators otherwise")
        args = parser.parse_args()
        run_exact = args.e

        runner = runEstimator(estimator=None, n_process=n_process, semivalue=None, semivalue_param=None,
                              game_func=gameTraining, game_args=None, nue_per_proc=nue_per_proc, nue_avg=nue_avg,
                              num_player=n_valued,nue_track_avg=nue_track_avg)

        for dataset in datasets:
            (X_valued, y_valued), (X_perf, y_perf), _, num_class = load_dataset(dataset=dataset, n_valued=n_valued,
                                                                                n_perf=n_perf)
            if dataset in ["MNIST", "FMNIST"]:
                arch, lr = "LeNet", 0.1
            else:
                arch, lr = "logistic", 1.0

            for metric in metrics:
                runner.game_args = dict(X_valued=X_valued, y_valued=y_valued, X_perf=X_perf, y_perf=y_perf, metric=metric,
                                        arch=arch, num_class=num_class, lr=lr)

                for key in semivalues.keys():
                    runner.semivalue = key[0]
                    runner.semivalue_param = key[1]

                    if run_exact:
                        path_cur = dir_format.format(metric, key[0], key[1], "exact_value")
                        value_saved = os.path.join(root, dataset, path_cur, "values.npz")
                        runner.estimator = "exact_value"
                        runner.file_prog = os.path.join(root, "prog_exact_values.txt")

                        with os_lock(value_saved, log=os.path.join(root, "log.txt")) as lock_state:
                            if lock_state:
                                values, _ = runner.run()
                                np.savez_compressed(value_saved, values=values)
                    else:
                        estimators = semivalues[key]
                        for estimator in estimators:
                            path_cur = os.path.join(root, dataset, dir_format.format(metric, key[0], key[1], estimator))
                            runner.estimator = estimator

                            for seed in seeds:
                                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                                runner.estimator_seed = seed
                                runner.file_prog = os.path.join(root, "prog_estimators.txt")

                                with os_lock(estimate_saved) as lock_state:
                                    if lock_state:
                                        _, estimates_traj = runner.run()
                                        np.savez_compressed(estimate_saved, estimates_traj=estimates_traj)
    except:
        traceback.print_exc()
        with open(os.path.join(root, "err.txt"), "a") as f:
            f.write("\n")
            traceback.print_exc(file=f)