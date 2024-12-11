import os
from compare_estimators import *
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator
from utils.funcs import export_legend

datasets = ["FMNIST"]
metrics = ["cross_entropy"]
list_semivalues = [('shapley', None), ('weighted_banzhaf', 0.5), ('beta_shapley', (4, 1)), ('beta_shapley', (1, 4)),
                   ('weighted_banzhaf', 0.2), ('weighted_banzhaf', 0.8)]

# see section Qualitative in https://matplotlib.org/stable/users/explain/colors/colormaps.html
clrs = sns.color_palette("Paired", 12)
clrs += sns.color_palette("Dark2", 8)
clrs += sns.color_palette('tab20', 20)
# clrs += sns.color_palette("tab20b", 20)
# clrs += sns.color_palette("tab20c", 20)
dict_color = dict(
    weighted_MSR=3,
    OFA_fixed=1,
    OFA_optimal=5,
    OFA_baseline=7,
    # WGELS_banzhaf=0,
    # WGELS_shapley=2,
    ARM_banzhaf=4,
    ARM_shapley=9,
    weighted_permutation=11,
    WSL_banzhaf=17,
    WSL=8,
    SHAP_IQ=32,
)
# dict_color_tmp = dict_color.copy()
# for key, value in dict_color_tmp.items():
#     if "_paired" in key:
#         dict_color.update({key[:-7] : value})
xticks = range(nue_track_avg, nue_avg + 1, nue_track_avg)

fig_format = os.path.join(root, "fig", "motivation;{};{};semivalue={}_{};{}.pdf")
os.makedirs(os.path.join(root, "fig"), exist_ok=True)

# plot the legend
dict_names = dict(
    OFA_optimal='OFA-A (ours)',
    OFA_fixed='OFA-S (ours)',
    WSL_banzhaf='WSL-Banzhaf',
    WSL='WSL-Shapley',
    # WGELS_banzhaf='GELS-B',
    # WGELS_shapley='GELS-S',
    ARM_banzhaf='ARM-Banzhaf',
    ARM_shapley='ARM-Shapley',
    weighted_MSR='MSR-Banzhaf',
    SHAP_IQ='SHAP-IQ',
    OFA_baseline='weightedSHAP',
    weighted_permutation='permutation-Shapley',
)

dict_shapley = dict(
    OFA_optimal='OFA_fixed',
    # WGELS_shapley='GELS',
    ARM_shapley='ARM',
    weighted_permutation='permutation',
    WSL='sampling_lift',
    SHAP_IQ='unbiased_kernelSHAP'
)

dict_banzhaf = dict(
    # WGELS_banzhaf='GELS',
    ARM_banzhaf='ARM',
    weighted_MSR='MSR',
    WSL_banzhaf='sampling_lift'
)

for estimator, label in dict_names.items():
    plt.plot([], [], label=label, color=clrs[dict_color[estimator]], linewidth=30)
legend = plt.legend(ncol=5, fontsize=100)
export_legend(legend, os.path.join(root, "fig", "legend.pdf"))
# plot the figures
for dataset in datasets:
    for metric in metrics:
        for key in list_semivalues:
            path_cur = dir_format.format(metric, key[0], key[1], "exact_value")
            value_saved = os.path.join(root, dataset, path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)


            error_dict = defaultdict(list)
            # correlation_dict = defaultdict(list)
            estimators = semivalues[key]
            for estimator in dict_color.keys():
                if key[0] == 'shapley' and estimator in dict_shapley:
                    est = dict_shapley[estimator]
                elif key[0] == 'weighted_banzhaf' and key[1] == 0.5 and estimator in dict_banzhaf:
                    est = dict_banzhaf[estimator]
                else:
                    est = estimator

                path_cur = os.path.join(root, dataset, dir_format.format(metric, key[0], key[1], est))

                estimates_collect = []
                for seed in seeds:
                    estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                    estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

                all_tmp = np.stack(estimates_collect)
                err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
                error_dict[estimator] = (err_tmp.mean(axis=0), err_tmp.std(axis=0))


            # sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots(figsize=(32, 24))
            ax.grid(linewidth=5)
            # plt.grid()

            for estimator, traj in error_dict.items():
                # if "paired" in estimator:
                #     plt.semilogy(xticks, traj, linestyle="--", c=clrs[dict_color[estimator]], linewidth=10)
                # else:
                #     plt.semilogy(xticks, traj, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
                plt.semilogy(xticks, traj[0], label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
                # ax.fill_between(xticks, traj[0] - traj[1], traj[0] + traj[1], alpha=0.2, facecolor=clrs[dict_color[estimator]])

            ax.tick_params(axis='x', labelsize=80)
            ax.tick_params(axis='y', labelsize=80)

            # to make the y-axis ticks of figure of beta(4,1) with cross_entropy on iris sparse
            # to avoid tick label overlapping
            # yticks = ax.get_yticks()
            # if len(yticks) == 6:
            #     ax.yaxis.set_major_locator(MaxNLocator(3))
            #     yticks = ax.yaxis.get_major_ticks()
            #     yticks[1].label1.set_visible(False)
            #     yticks[2].label1.set_visible(False)
            #     ygridlines = ax.get_ygridlines()
            #     ygridlines[1].set_visible(False)
            #     ygridlines[2].set_visible(False)

            plt.xlabel("#utility evaluations per player", fontsize=100)
            plt.ylabel("relative difference", fontsize=100)
            fig_saved = fig_format.format(dataset, metric, key[0], key[1], "error")
            plt.savefig(fig_saved, bbox_inches='tight')
            plt.close(fig)




