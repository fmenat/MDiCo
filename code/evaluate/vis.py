import matplotlib.pyplot as plt
import numpy as np


def plot_conf_matrix(ax, cf_matrix, add_title=""):
    import seaborn as sns
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix,axis=1, keepdims=True),
		annot=True, fmt='.2%', cmap='Blues',  vmin=0, vmax=1, cbar=False)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.set_title(f"Confusion {add_title}")


def plot_prob_dist_bin(ax, y_pred_prob, y_true, add_title=""):
    mask_non_crop = y_true == 0
    mask_crop = y_true == 1
    binwidth = 0.04
    bins = np.arange(0, 1+ binwidth, binwidth)
    ax.hist(y_pred_prob[mask_non_crop,1], label="Negative", alpha=0.5,bins=bins,edgecolor='white', linewidth=1.2)
    ax.hist(y_pred_prob[mask_crop,1], label="Target Crop", alpha=0.5,bins=bins, edgecolor='black', linewidth=1.2)
    ax.set_xlim(0,1)
    ax.axvline(0.5, ls="dashed", lw=2, label="Probability Threshold", color="black")
    ax.set_title(f"Histogram of the predicted probability {add_title}")
    ax.legend(loc="upper center")
    ax.set_xlabel("Target Crop Probability")
    ax.set_ylabel("Count")


def plot_dist_bin(ax, y_pred_cont, y_true, add_title=""):
    binwidth = 5 if np.max(y_true) > 50  else 0.5
    bins = np.arange(0, np.max(y_true)+ binwidth, binwidth)
    ax.hist(y_true, label="Ground Truth", alpha=0.6, bins=bins, edgecolor='black', linewidth=1.2)
    ax.hist(y_pred_cont, label="Prediction", alpha=0.35, bins=bins,edgecolor='black', linewidth=1.2 )
    ax.set_title(f"Histogram of target values {add_title}")
    ax.legend(loc="upper right")
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    ax.set_xlim(0)

def plot_true_vs_pred(ax, y_pred_cont, y_true, add_title=""):
    y = np.arange(np.min(y_true), np.max(y_true))
    ax.plot(y, y, "-", color="red")
    ax.scatter(y_true, y_pred_cont, marker="o", edgecolors='black', s=30)
    ax.set_title(f"Prediction vs ground truth {add_title}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")

def plot_comparison_radar(ax, metrics, metrics_names, categories, add_title=""):
    angles = np.linspace(0, 2*np.pi, len(metrics[0]), endpoint=False).tolist() #in radans
    angles += angles[:1] # repeat first angle to close poly    # plot

    for metric, metric_names in zip(metrics, metrics_names):
        ax.plot(angles, [*metric, metric[0]], linestyle="--", marker="o", label=metric_names)
        ax.fill(angles, [*metric, metric[0]], alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_rlim(0,1)
    ax.set_rticks(np.linspace(0,1, 6))
    ax.set_rlabel_position(90*angles[1]/np.pi)
    ax.grid(True)
    ax.set_title(f"Methods comparison {add_title}")
    ax.legend(loc="lower right")


def plot_attention(ax, means_ ,stds_, view_names = [], add_title=""):
    if len(view_names) == 0:
        view_names = [f"S{str(v)}" for v in np.arange(len(means_))]
    ax.plot(means_, "o-")
    ax.fill_between(np.arange(len(view_names)), means_-stds_, means_+stds_, alpha=0.5)
    ax.set_xticks(np.arange(len(view_names)))
    ax.set_xticklabels(view_names)
    ax.set_ylim(0,1)
    ax.set_title("Average pixel attention "+add_title)

def plot_col_feat_result(ax, df, col, marker, add_title="", **args_plot):
    ax.plot(df[col].values, label = col, marker = marker, **args_plot)
    ax.set_title(col+add_title)
    ax.set_ylim(-0.05,1.05)
    ax.set_xlabel("features-axis")
    ax.set_xticks(np.arange(len(df[col])))

def plot_col_feat_mean_std(axx, df_mean,df_std, makers_plot, **args_plot):
    for i, col in enumerate(df_mean.columns):
        axx[0,0].plot(df_mean[col], label = col, marker = makers_plot[i], **args_plot)
        axx[0,1].plot(df_std[col], label = col, marker = makers_plot[i], **args_plot)
    for i in range(axx.shape[1]):
        axx[0,i].set_ylim(-0.05,1.05)
        axx[0,i].set_xlabel("features-axis")
        axx[0,i].legend()
    axx[0,0].set_ylabel("MEAN over runs")
    axx[0,1].set_ylabel("STD over runs")
