import os
from compare_estimators import root, dir_format
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator
from utils.funcs import export_legend
import numpy as np

datasets = ["FMNIST", "MNIST", "iris"]
metrics = ["cross_entropy"]
game_param = [64, 128, 256]
index_take = np.arange(100)

# see section Qualitative in https://matplotlib.org/stable/users/explain/colors/colormaps.html
clrs = sns.color_palette("Paired", 12)
clrs += sns.color_palette("Dark2", 8)
clrs += sns.color_palette('tab20', 20)
dict_color = dict(
    OFA_optimal=5,
    OFA_baseline=7,
    weighted_permutation=11,
    WSL=8,
    SHAP_IQ=32,
)

os.makedirs(os.path.join(root, "fig"), exist_ok=True)

# plot the legend
dict_names = dict(
    OFA_optimal='OFA-A (ours)',
    WSL='WSL-Shapley',
    SHAP_IQ='SHAP-IQ',
    OFA_baseline='weightedSHAP',
    weighted_permutation='permutation-Shapley',
)

dict_shapley = dict(
    OFA_optimal='OFA_fixed',
    weighted_permutation='permutation',
    WSL='sampling_lift',
    SHAP_IQ='unbiased_kernelSHAP'
)

list_bs = [(16, 1), (4, 1), (1, 1), (1, 4), (1, 16), (16, 4), (2, 2), (8, 8), (4, 16)]
list_wb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for estimator, label in dict_names.items():
    plt.plot([], [], label=label, color=clrs[dict_color[estimator]], linewidth=30)
legend = plt.legend(ncol=5, fontsize=100)
export_legend(legend, os.path.join(root, "fig", "legend_ofa.pdf"))



# plot the figures
for dataset in datasets:
    for metric in metrics:
        aucc_dict = dict()

        for i, param in enumerate(list_bs):
            if param == (1, 1):
                semivalue = 'shapley'
                semivalue_param = None
            else:
                semivalue = 'beta_shapley'
                semivalue_param = param

            path_cur = dir_format.format(metric, semivalue, semivalue_param, "exact_value")
            value_saved = os.path.join(root, dataset, path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)

            for estimator in dict_names.keys():
                if semivalue == 'shapley' and estimator in dict_shapley:
                    est = dict_shapley[estimator]
                else:
                    est = estimator

                if estimator not in aucc_dict:
                    aucc_dict[estimator] = np.empty((30, 9), dtype=np.float64)

                path_cur = os.path.join(root, dataset, dir_format.format(metric, semivalue, semivalue_param, est))
                estimates_collect = []
                for seed in np.arange(30):
                    estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                    estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

                all_tmp = np.stack(estimates_collect)
                err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
                aucc_dict[estimator][:, i] = err_tmp[:, index_take].mean(axis=1)


        # sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(32, 24))
        ax.grid(linewidth=5)
        plt.xticks(np.arange(9), [str(e) for e in list_bs])
        xticklabels = ax.get_xticklabels()
        for i in range(9):
            xticklabels[i].set_rotation(-90)

        for estimator, traj in aucc_dict.items():
            traj_mean = traj.mean(axis=0)
            traj_std = traj.std(axis=0)
            plt.semilogy(np.arange(9), traj_mean, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
            ax.fill_between(np.arange(9), traj_mean - traj_std, traj_mean + traj_std, alpha=0.3, facecolor=clrs[dict_color[estimator]])


        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)


        plt.xlabel("Beta Shapley values", fontsize=100)
        plt.ylabel("AUCC", fontsize=100)
        fig_saved = os.path.join(root, "fig", f"ofa;{dataset};{metric};beta_shapley.pdf")
        plt.savefig(fig_saved, bbox_inches='tight')
        plt.close(fig)


        semivalue = 'weighted_banzhaf'
        for i, param in enumerate(list_wb):
            semivalue_param = param

            path_cur = dir_format.format(metric, semivalue, semivalue_param, "exact_value")
            value_saved = os.path.join(root, dataset, path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)

            for estimator in dict_names.keys():
                if estimator not in aucc_dict:
                    aucc_dict[estimator] = np.empty((30, 9), dtype=np.float64)

                path_cur = os.path.join(root, dataset, dir_format.format(metric, semivalue, semivalue_param, estimator))
                estimates_collect = []
                for seed in np.arange(30):
                    estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                    estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

                all_tmp = np.stack(estimates_collect)
                err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
                aucc_dict[estimator][:, i] = err_tmp[:, index_take].mean(axis=1)


        # sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(32, 24))
        ax.grid(linewidth=5)
        plt.xticks(np.arange(9), [str(e) for e in list_wb])

        for estimator, traj in aucc_dict.items():
            traj_mean = traj.mean(axis=0)
            traj_std = traj.std(axis=0)
            plt.semilogy(np.arange(9), traj_mean, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
            ax.fill_between(np.arange(9), traj_mean - traj_std, traj_mean + traj_std, alpha=0.3,
                            facecolor=clrs[dict_color[estimator]])


        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)


        plt.xlabel("weighted Banzhaf values", fontsize=100)
        plt.ylabel("AUCC", fontsize=100)
        fig_saved = os.path.join(root, "fig", f"ofa;{dataset};{metric};weighted_banzhaf.pdf")
        plt.savefig(fig_saved, bbox_inches='tight')
        plt.close(fig)


root = os.path.join("exp", "compare_estimators;nue_avg=2000")
dir_format = "semivalue={}_{};estimator={}"
for game in game_param:
    aucc_dict = dict()

    for i, param in enumerate(list_bs):
        if param == (1, 1):
            semivalue = 'shapley'
            semivalue_param = None
        else:
            semivalue = 'beta_shapley'
            semivalue_param = param

        path_cur = dir_format.format(semivalue, semivalue_param, "exact_value")
        value_saved = os.path.join(root, f"n_player={game};n_unanimity={game**2}", path_cur, "values.npz")
        values_exact = np.load(value_saved)["values"]
        norm_exact = np.linalg.norm(values_exact)

        for estimator in dict_names.keys():
            if semivalue == 'shapley' and estimator in dict_shapley:
                est = dict_shapley[estimator]
            else:
                est = estimator

            if estimator not in aucc_dict:
                aucc_dict[estimator] = np.empty((30, 9), dtype=np.float64)

            path_cur = os.path.join(root, f"n_player={game};n_unanimity={game**2}",
                                    dir_format.format(semivalue, semivalue_param, est))
            estimates_collect = []
            for seed in np.arange(30):
                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

            all_tmp = np.stack(estimates_collect)
            err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
            aucc_dict[estimator][:, i] = err_tmp[:, index_take].mean(axis=1)

    # sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(32, 24))
    ax.grid(linewidth=5)
    plt.xticks(np.arange(9), [str(e) for e in list_bs])
    xticklabels = ax.get_xticklabels()
    for i in range(9):
        xticklabels[i].set_rotation(-90)

    for estimator, traj in aucc_dict.items():
        traj_mean = traj.mean(axis=0)
        traj_std = traj.std(axis=0)
        plt.semilogy(np.arange(9), traj_mean, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
        ax.fill_between(np.arange(9), traj_mean - traj_std, traj_mean + traj_std, alpha=0.3,
                        facecolor=clrs[dict_color[estimator]])

    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)

    plt.xlabel("Beta Shapley values", fontsize=100)
    plt.ylabel("AUCC", fontsize=100)
    fig_saved = os.path.join(root, "fig", f"ofa;sou={game};beta_shapley.pdf")
    plt.savefig(fig_saved, bbox_inches='tight')
    plt.close(fig)

    semivalue = 'weighted_banzhaf'
    for i, param in enumerate(list_wb):
        semivalue_param = param

        path_cur = dir_format.format(semivalue, semivalue_param, "exact_value")
        value_saved = os.path.join(root, f"n_player={game};n_unanimity={game**2}", path_cur, "values.npz")
        values_exact = np.load(value_saved)["values"]
        norm_exact = np.linalg.norm(values_exact)

        for estimator in dict_names.keys():
            if estimator not in aucc_dict:
                aucc_dict[estimator] = np.empty((30, 9), dtype=np.float64)

            path_cur = os.path.join(root, f"n_player={game};n_unanimity={game**2}",
                                    dir_format.format(semivalue, semivalue_param, estimator))
            estimates_collect = []
            for seed in np.arange(30):
                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

            all_tmp = np.stack(estimates_collect)
            err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
            aucc_dict[estimator][:, i] = err_tmp[:, index_take].mean(axis=1)

    # sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(32, 24))
    ax.grid(linewidth=5)
    plt.xticks(np.arange(9), [str(e) for e in list_wb])

    for estimator, traj in aucc_dict.items():
        traj_mean = traj.mean(axis=0)
        traj_std = traj.std(axis=0)
        plt.semilogy(np.arange(9), traj_mean, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
        ax.fill_between(np.arange(9), traj_mean - traj_std, traj_mean + traj_std, alpha=0.3,
                        facecolor=clrs[dict_color[estimator]])

    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)

    plt.xlabel("weighted Banzhaf values", fontsize=100)
    if game == 128:
        plt.ylabel("AUCC", fontsize=100, labelpad=150)
    else:
        plt.ylabel("AUCC", fontsize=100)
    fig_saved = os.path.join(root, "fig", f"ofa;sou={game};weighted_banzhaf.pdf")
    plt.savefig(fig_saved, bbox_inches='tight')
    plt.close(fig)



