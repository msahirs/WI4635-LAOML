import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

def get_data_from_file(filename):
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

def batch_size_1plot(df):
    g = sns.catplot(
        data=df, 
        x="batch_size%", 
        y="test_performance", 
        col="alpha",
        col_wrap=3,
        legend=True,
        legend_out=False,
        sharex=False,
        hue="reg",
        palette=sns.color_palette("deep", 5)
    )
    plt.setp(g._legend.get_title(), fontsize=20)
    plt.setp(g._legend.get_texts(), fontsize=50)
    g.set_titles("{col_name} {col_var}")
    g.set(ylim=(.9, 1))
    sns.move_legend(g, "upper left", bbox_to_anchor=(.75, .40), frameon=False)
    g.savefig("batch_test.png")

    g = sns.catplot(
        data=df, 
        x="batch_size%", 
        y="elapsed_time", 
        col="alpha",
        col_wrap=3,
        legend=True,
        legend_out=False,
        sharex=False,
        hue="reg",
        palette=sns.color_palette("deep", 5)
    )
    plt.setp(g._legend.get_title(), fontsize=20)
    plt.setp(g._legend.get_texts(), fontsize=50)
    g.set_titles("{col_name} {col_var}")
    # g.set(ylim=(.9, 1))
    # g.despine(left=True)
    sns.move_legend(g, "upper left", bbox_to_anchor=(.75, .40), frameon=False)
    g.savefig("batch_test_timing.png")


def batch_size_convergence(df):
    data = df.explode("convergence").reset_index().rename(columns={'index' : 'iteration'})
    data["iteration"] = data.groupby('iteration').cumcount()
    g = sns.relplot(
        data=data, 
        x="iteration", 
        y="convergence", 
        col="alpha",
        col_wrap=3,
        hue="batch_size%", 
        style="batch_size%",
        palette=sns.color_palette("deep", 5),
        kind="line",
        # errorbar=ci
    )
    g.set_titles("{col_name} {col_var}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(.75, .40), frameon=False)
    g.savefig("convergence_error.png")

def split_batch_size(df):
    g = sns.catplot(
        data=df, 
        x="reg", 
        y="test_performance", 
        col="batch_size%",
        hue="alpha",

        col_wrap=3,
        legend=True,
        legend_out=False,
        sharex=False,
        palette=sns.color_palette("deep", 5)
    )
    plt.setp(g._legend.get_title(), fontsize=20)
    plt.setp(g._legend.get_texts(), fontsize=50)
    g.set_titles("{col_name} {col_var}")
    g.set(ylim=(.9, 1))
    sns.move_legend(g, "upper left", bbox_to_anchor=(.75, .40), frameon=False)
    g.savefig("batch_test.png")

def full_convergence(df):
    g = sns.catplot(
        data=df, 
        x="reg", 
        y="test_performance", 
        col="batch_size%",
        hue="alpha",

        col_wrap=3,
        legend=True,
        legend_out=False,
        sharex=False,
        palette=sns.color_palette("deep", 5)
    )
    plt.setp(g._legend.get_title(), fontsize=20)
    plt.setp(g._legend.get_texts(), fontsize=50)
    g.set_titles("{col_name} {col_var}")
    g.set(ylim=(.9, 1))
    sns.move_legend(g, "upper left", bbox_to_anchor=(.75, .40), frameon=False)
    g.savefig("batch_test.png")

if __name__ == "__main__":
    data = get_data_from_file("batch_test_data")
    df = pd.DataFrame(data)[-125:]
    # batch_size_1plot(df)
    split_batch_size(df)
