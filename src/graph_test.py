import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

def get_data_from_file(filename):
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

def batch_size_impact(df):
    group_keys = ["alpha", "reg"]
    distinct_counts = [len(df[k].value_counts()) for k in group_keys]
    df_grouped = df.groupby(group_keys)
    f, axs = plt.subplots(*distinct_counts, figsize=(15, 15))
    for x, ax_id in zip(df_grouped, np.ndindex(axs.shape)):
        
        sns.scatterplot(
                    x="batch_size%", 
                    y="test_performance",
                    data=x[1], 
                    ax=axs[ax_id]
                    )
        axs[ax_id].set_title(", ".join([f"{k}={val}" for k, val in zip(group_keys, x[0])]))
        axs[ax_id].set(ylim=(0.92, 0.98))
        if ax_id[1] != 0:
            axs[ax_id].get_yaxis().set_ticklabels([])
            axs[ax_id].set(ylabel=None)
        
        if ax_id[0] != distinct_counts[0] - 1:
            axs[ax_id].get_xaxis().set_ticklabels([])
            axs[ax_id].set(xlabel=None)
    
    f.savefig("batch_test.png")

def batch_size_1plot(df):
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set(ylim=(0.9, 0.98))
    sns.scatterplot(
                    x="reg", 
                    y="test_performance",
                    hue="alpha",
                    size="batch_size%",
                    data=df, 
                    ax=ax
                    )
    f.savefig("batch_test.png")

def batch_size_convergence(df):
    group_keys = ["alpha", "reg"]
    distinct_counts = [len(df[k].value_counts()) for k in group_keys]

    df_grouped = df.groupby(group_keys)
    f, axs = plt.subplots(*distinct_counts, figsize=(15, 15))
    for x, ax_id in zip(df_grouped, np.ndindex(axs.shape)):
        data = x[1].copy().explode("convergence").reset_index().rename(columns={'index' : 'iteration'})
        data["iteration"] = data.groupby('iteration').cumcount()
        print(data["convergence"])
        sns.lineplot(
            data=data,
            x="iteration",
            y="convergence",
            hue="batch_size%",
            palette=sns.color_palette("deep", 5),
            ax=axs[ax_id]
            )
        
        axs[ax_id].set_title(", ".join([f"{k}={val}" for k, val in zip(group_keys, x[0])]))
        axs[ax_id].set(ylim=(0.0, 1.0))
        if ax_id[1] != 0:
            axs[ax_id].get_yaxis().set_ticklabels([])
            axs[ax_id].set(ylabel=None)
        
        if ax_id[0] != distinct_counts[0] - 1:
            axs[ax_id].get_xaxis().set_ticklabels([])
            axs[ax_id].set(xlabel=None)
    
    f.savefig("batch_test.png")


if __name__ == "__main__":
    data = get_data_from_file("batch_test_data")
    df = pd.DataFrame(data)
    # df = df.round(7)
    # batch_size_impact(df)
    batch_size_1plot(df)
    # batch_size_convergence(df)
