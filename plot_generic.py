import os
from compare_estimators import root, dir_format
from collections import defaultdict
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
clrs += sns.color_palette("Accent", 8)
clrs += sns.color_palette("Set1", 9)
clrs += sns.color_palette('tab20', 20)
dict_color = dict(
    OFA_optimal=5,
    OFA_fixed=1,
    GELS=0,
    GELS_shapley=7,
    sampling_lift=2,
    kernelSHAP_paired=3,
    GELS_shapley_paired=4,
    GELS_paired=17,
    ARM=8,
    complement=9,
    improved_AME=45,
    MSR=11,
    test_leverage_paired=11,
    modified_leverage_paired=10
)
os.makedirs(os.path.join(root, "fig"), exist_ok=True)

# plot the legend
dict_names = dict(
    OFA_optimal='OFA-A (ours)',
    OFA_fixed='OFA-S (ours)',
    GELS='GELS',
    GELS_shapley='GELS-Shapley',
    sampling_lift='sampling lift',
    kernelSHAP_paired='kernelSHAP',
    GELS_shapley_paired='unbiased kernelSHAP',
    GELS_paired='group testing',
    ARM='ARM',
    complement='complement',
    improved_AME='AME',
    MSR='MSR',
)

seeds = dict(FMNIST=14, MNIST=25, iris=14)
seeds.update({64: 8, 128: 8, 256: 14})

list_bs = [(16, 1), (4, 1), (1, 1), (1, 4), (1, 16), (16, 4), (2, 2), (8, 8), (4, 16)]
list_wb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for estimator, label in dict_names.items():
    plt.plot([], [], label=label, color=clrs[dict_color[estimator]], linewidth=30)
legend = plt.legend(ncol=6, fontsize=100)
export_legend(legend, os.path.join(root, "fig", "legend_generic.pdf"))


# plot the figures
for dataset in datasets:
    for metric in metrics:

        # sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(32, 24))
        ax.grid(linewidth=5)
        plt.xticks(np.arange(9), [str(e) for e in list_bs])
        xticklabels = ax.get_xticklabels()
        for i in range(9):
            xticklabels[i].set_rotation(-90)
        plt.xlabel("Beta Shapley values", fontsize=100)
        plt.ylabel("AUCC", fontsize=100)
        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)

        for estimator in ["OFA_optimal", "OFA_fixed", "GELS", "ARM", 'sampling_lift', 'improved_AME']:
            aucc = np.empty((30, 9), dtype=np.float64)
            for i, param in enumerate(list_bs):
                if param == (1, 1):
                    semivalue = 'shapley'
                    semivalue_param = None
                else:
                    semivalue = 'beta_shapley'
                    semivalue_param = param

                if estimator == 'OFA_optimal' and param == (1, 1):
                    est = 'OFA_fixed'
                else:
                    est = estimator

                path_cur = dir_format.format(metric, semivalue, semivalue_param, "exact_value")
                value_saved = os.path.join(root, dataset, path_cur, "values.npz")
                values_exact = np.load(value_saved)["values"]
                norm_exact = np.linalg.norm(values_exact)

                path_cur = os.path.join(root, dataset, dir_format.format(metric, semivalue, semivalue_param, est))
                estimates_collect = []
                for seed in np.arange(30):
                    estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                    estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

                all_tmp = np.stack(estimates_collect)
                err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
                aucc[:, i] = err_tmp[:, index_take].mean(axis=1)

            aucc_mean = aucc.mean(axis=0)
            aucc_std = aucc.std(axis=0)

            if estimator == "improved_AME":
                if dataset == "iris":
                    tmp = np.delete(aucc[:, 0], 8)
                    aucc_mean[0] = tmp.mean()
                    aucc_std[0] = tmp.std()
                plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10, linestyle='--')
            else:
                plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10)
            ax.fill_between(np.arange(9), aucc_mean - aucc_std, aucc_mean + aucc_std, alpha=0.3,
                            facecolor=clrs[dict_color[estimator]])


        estimator = 'AME'
        aucc = np.empty((30, 4), dtype=np.float64)
        for i, param in enumerate(list_bs[5:]):
            path_cur = dir_format.format(metric, 'beta_shapley', param, "exact_value")
            value_saved = os.path.join(root, dataset, path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)

            path_cur = os.path.join(root, dataset, dir_format.format(metric, 'beta_shapley', param, estimator))
            estimates_collect = []
            for seed in np.arange(30):
                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

            all_tmp = np.stack(estimates_collect)
            err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
            aucc[:, i] = err_tmp[:, index_take].mean(axis=1)
        aucc_mean = aucc.mean(axis=0)
        aucc_std = aucc.std(axis=0)
        plt.semilogy(np.arange(5, 9), aucc_mean, c=clrs[dict_color["improved_AME"]], linewidth=10)
        ax.fill_between(np.arange(5, 9), aucc_mean - aucc_std, aucc_mean + aucc_std, alpha=0.3,
                        facecolor=clrs[dict_color["improved_AME"]])

        np.random.seed(seeds[dataset])
        for estimator in ["GELS_shapley", 'kernelSHAP_paired', 'GELS_shapley_paired', 'GELS_paired', 'complement', 'test_leverage_paired', 'modified_leverage_paired']:
            path_cur = dir_format.format(metric, 'shapley', None, "exact_value")
            value_saved = os.path.join(root, dataset, path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)

            path_cur = os.path.join(root, dataset, dir_format.format(metric, 'shapley', None, estimator))
            estimates_collect = []
            for seed in np.arange(30):
                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

            all_tmp = np.stack(estimates_collect)
            err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
            aucc = err_tmp[:, index_take].mean(axis=1)

            offset = np.random.uniform(-0.2, 0.2)
            plt.errorbar(2+offset, aucc.mean(), yerr=aucc.std(), fmt='o', markersize=30, color=clrs[dict_color[estimator]],
                         ecolor=clrs[dict_color[estimator]], capsize=15, elinewidth=5, capthick=5)


        fig_saved = os.path.join(root, "fig", f"generic,{dataset},{metric},beta_shapley.pdf")
        plt.savefig(fig_saved, bbox_inches='tight')
        plt.close(fig)

        # sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(32, 24))
        ax.grid(linewidth=5)
        plt.xticks(np.arange(9), [str(e) for e in list_wb])
        plt.xlabel("weighted Banzhaf values", fontsize=100)
        plt.ylabel("AUCC", fontsize=100)
        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)


        for estimator in ["OFA_optimal", "OFA_fixed", "GELS", "ARM", 'sampling_lift', 'MSR', 'AME', 'improved_AME']:
            aucc = np.empty((30, 9), dtype=np.float64)
            for i, param in enumerate(list_wb):
                path_cur = dir_format.format(metric, 'weighted_banzhaf', param, "exact_value")
                value_saved = os.path.join(root, dataset, path_cur, "values.npz")
                values_exact = np.load(value_saved)["values"]
                norm_exact = np.linalg.norm(values_exact)

                path_cur = os.path.join(root, dataset, dir_format.format(metric, 'weighted_banzhaf', param, estimator))
                estimates_collect = []
                for seed in np.arange(30):
                    estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                    estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

                all_tmp = np.stack(estimates_collect)
                err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
                aucc[:, i] = err_tmp[:, index_take].mean(axis=1)
            aucc_mean = aucc.mean(axis=0)
            aucc_std = aucc.std(axis=0)
            if estimator == "improved_AME":
                plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10, linestyle='--')
            else:
                if estimator == "AME":
                    estimator = "improved_AME"
                plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10)
            ax.fill_between(np.arange(9), aucc_mean - aucc_std, aucc_mean + aucc_std, alpha=0.3,
                            facecolor=clrs[dict_color[estimator]])

        fig_saved = os.path.join(root, "fig", f"generic,{dataset},{metric},weighted_banzhaf.pdf")
        plt.savefig(fig_saved, bbox_inches='tight')
        plt.close(fig)





root = os.path.join("exp", "compare_estimators;nue_avg=2000")
dir_format = "semivalue={}_{};estimator={}"
for game in game_param:

    # sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(32, 24))
    ax.grid(linewidth=5)
    plt.xticks(np.arange(9), [str(e) for e in list_bs])
    xticklabels = ax.get_xticklabels()
    for i in range(9):
        xticklabels[i].set_rotation(-90)
    plt.xlabel("Beta Shapley values", fontsize=100)
    plt.ylabel("AUCC", fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)

    for estimator in ["OFA_optimal", "OFA_fixed", "GELS", "ARM", 'sampling_lift', "improved_AME"]:
        aucc = np.empty((30, 9), dtype=np.float64)
        for i, param in enumerate(list_bs):
            if param == (1, 1):
                semivalue = 'shapley'
                semivalue_param = None
            else:
                semivalue = 'beta_shapley'
                semivalue_param = param

            if estimator == 'OFA_optimal' and param == (1, 1):
                est = 'OFA_fixed'
            else:
                est = estimator

            path_cur = dir_format.format(semivalue, semivalue_param, "exact_value")
            value_saved = os.path.join(root, f"n_player={game};n_unanimity={game**2}", path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)

            path_cur = os.path.join(root, f"n_player={game};n_unanimity={game**2}",
                                    dir_format.format(semivalue, semivalue_param, est))
            estimates_collect = []
            for seed in np.arange(30):
                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

            all_tmp = np.stack(estimates_collect)
            err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
            aucc[:, i] = err_tmp[:, index_take].mean(axis=1)

        aucc_mean = aucc.mean(axis=0)
        aucc_std = aucc.std(axis=0)
        if estimator == "improved_AME":
            if game == 64:
                tmp = np.delete(aucc[:, 3], -6)
                aucc_mean[3] = tmp.mean()
                aucc_std[3] = tmp.std()
            plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10, linestyle='--')
        else:
            plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10)
        ax.fill_between(np.arange(9), aucc_mean - aucc_std, aucc_mean + aucc_std, alpha=0.3,
                        facecolor=clrs[dict_color[estimator]])


    estimator = 'AME'
    aucc = np.empty((30, 4), dtype=np.float64)
    for i, param in enumerate(list_bs[5:]):
        path_cur = dir_format.format('beta_shapley', param, "exact_value")
        value_saved = os.path.join(root, f"n_player={game};n_unanimity={game**2}", path_cur, "values.npz")
        values_exact = np.load(value_saved)["values"]
        norm_exact = np.linalg.norm(values_exact)

        path_cur = os.path.join(root, f"n_player={game};n_unanimity={game**2}",
                                dir_format.format('beta_shapley', param, estimator))
        estimates_collect = []
        for seed in np.arange(30):
            estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
            estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

        all_tmp = np.stack(estimates_collect)
        err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
        aucc[:, i] = err_tmp[:, index_take].mean(axis=1)
    aucc_mean = aucc.mean(axis=0)
    aucc_std = aucc.std(axis=0)
    plt.semilogy(np.arange(5, 9), aucc_mean, c=clrs[dict_color["improved_AME"]], linewidth=10)
    ax.fill_between(np.arange(5, 9), aucc_mean - aucc_std, aucc_mean + aucc_std, alpha=0.3,
                    facecolor=clrs[dict_color["improved_AME"]])

    np.random.seed(seeds[game])
    for estimator in ["GELS_shapley", 'kernelSHAP_paired', 'GELS_shapley_paired', 'GELS_paired', 'complement', 'test_leverage_paired', 'modified_leverage_paired']:
        path_cur = dir_format.format('shapley', None, "exact_value")
        value_saved = os.path.join(root, f"n_player={game};n_unanimity={game**2}", path_cur, "values.npz")
        values_exact = np.load(value_saved)["values"]
        norm_exact = np.linalg.norm(values_exact)

        path_cur = os.path.join(root, f"n_player={game};n_unanimity={game**2}",
                                dir_format.format('shapley', None, estimator))
        estimates_collect = []
        for seed in np.arange(30):
            estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
            estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

        all_tmp = np.stack(estimates_collect)
        err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
        aucc = err_tmp[:, index_take].mean(axis=1)

        offset = np.random.uniform(-0.2, 0.2)
        plt.errorbar(2 + offset, aucc.mean(), yerr=aucc.std(), fmt='o', markersize=30,
                     color=clrs[dict_color[estimator]],
                     ecolor=clrs[dict_color[estimator]], capsize=18, elinewidth=5, capthick=5)


    fig_saved = os.path.join(root, "fig", f"generic,sou={game},beta_shapley.pdf")
    plt.savefig(fig_saved, bbox_inches='tight')
    plt.close(fig)

    # sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(32, 24))
    ax.grid(linewidth=5)
    plt.xticks(np.arange(9), [str(e) for e in list_wb])
    plt.xlabel("weighted Banzhaf values", fontsize=100)
    plt.ylabel("AUCC", fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)

    for estimator in ["OFA_optimal", "OFA_fixed", "GELS", "ARM", 'sampling_lift', 'MSR', 'AME', 'improved_AME']:
        aucc = np.empty((30, 9), dtype=np.float64)
        for i, param in enumerate(list_wb):
            path_cur = dir_format.format('weighted_banzhaf', param, "exact_value")
            value_saved = os.path.join(root, f"n_player={game};n_unanimity={game**2}", path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)

            path_cur = os.path.join(root, f"n_player={game};n_unanimity={game**2}",
                                    dir_format.format('weighted_banzhaf', param, estimator))
            estimates_collect = []
            for seed in np.arange(30):
                estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

            all_tmp = np.stack(estimates_collect)
            err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
            aucc[:, i] = err_tmp[:, index_take].mean(axis=1)
        aucc_mean = aucc.mean(axis=0)
        aucc_std = aucc.std(axis=0)
        if estimator == "improved_AME":
            plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10, linestyle='--')
        else:
            if estimator == "AME":
                estimator = "improved_AME"
            plt.semilogy(np.arange(9), aucc_mean, c=clrs[dict_color[estimator]], linewidth=10)
        ax.fill_between(np.arange(9), aucc_mean - aucc_std, aucc_mean + aucc_std, alpha=0.3,
                        facecolor=clrs[dict_color[estimator]])

    fig_saved = os.path.join(root, "fig", f"generic,sou={game},weighted_banzhaf.pdf")
    plt.savefig(fig_saved, bbox_inches='tight')
    plt.close(fig)


