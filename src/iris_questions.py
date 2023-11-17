import iris_classification as ic
import matplotlib.pyplot as plt
import numpy as np

# For copying and archiving files
import shutil

# Use TeX backend if available, else standard text engine
plt.rcParams['text.usetex']  = True \
    if shutil.which('latex') else False



iris_data_loc = './data/iris.csv' # File name of IRIS dataset
data_array = ic.extract_dataset(iris_data_loc)

def part_c(frac = 0, weight_samples = 100):

    """ Sahir explanation for having samples generated which give better weights than
    the least-squares solution is the additon of noise to the weights can also act
    as means of regularisation, such as by reducing overfititng or the effects of outliers
    in some instances. It can be difficult to predict where to add the noise however, i.e.
    in which weights to add noise (and with which parameters/distributions)"""

    # generate data and outcome
    train_X, train_y, test_X, test_y = ic.split_data_rnd(data_array, fraction=frac)

    n_corr, w_ref = ic.cluster_dataset_test(test_X,test_y,test_X,test_y)

    print(n_corr)


    scale_list = np.linspace(0,0.5,10000).tolist()
    highest_correct = 0
    high_scale = 0
    num_higher = []

    for scale in scale_list:

        w_vars_uni = w_ref + (np.random.uniform(size = (weight_samples,test_X.shape[1])) - 0.5) * scale
        
        corr_higher = 0

        for i in w_vars_uni:
            a = ic.calculate_correct(test_X,test_y,i)

            if a > n_corr:
                corr_higher +=1
            
        if corr_higher > highest_correct:
            high_scale = scale
            highest_correct = corr_higher
        num_higher.append(corr_higher)

    print("highest number of more correct", highest_correct)
    print("scale with highest accuracy:",high_scale)

    plt.figure(figsize = (12, 8))

    plt.plot(scale_list,num_higher, color = 'r', alpha = 0.8)
    
    plt.xlabel("Scale of Noise",fontsize=16)
    plt.ylabel(f"No. of Weight Vectors",fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13) 

    plt.title(f"No. of Weight vectors (out of {weight_samples}) better than LSE",fontsize=20)

    plt.tight_layout()
    # plt.legend()
    plt.savefig("iris_figures/weight_noise", dpi = 300)


    # print(f"In {weight_samples} random weight samples, {corr_higher} higher than least-squares solution")

    


def main():
    part_c()


if __name__ == "__main__":
    main()




