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

def part_d_i(frac = 0.5):

    plt.figure(figsize = (12, 8))


    train_X, train_y, test_X, test_y = ic.split_data_rnd(data_array, fraction=frac)

    reg = 5e-2

    no_correct, weight_vec = ic.cluster_dataset_test(train_X, train_y, test_X, test_y)
    w_cg = ic.tikhonov_cg_lse(train_X,train_y, reg_param= reg)
    w_qr = ic.tikhonov_qr_lse(train_X,train_y, reg_param= reg)
    w_bd = ic.tikhonov_bold_driver(train_X, train_y, reg_param=reg)

    print("Weights of cg:", w_cg)
    print(f"Number correct (out of {test_y.size})", ic.calculate_correct(test_X, test_y, w_cg))
    print("Weights of qr:", w_qr)
    print(f"Number correct (out of {test_y.size})", ic.calculate_correct(test_X, test_y, w_qr))
    print("weights of bd:", w_bd)
    print(f"Number correct (out of {test_y.size})", ic.calculate_correct(test_X, test_y, w_bd))
    print("Weights of hyperplane clustering:", weight_vec)
    print(f"Number correct (out of {test_y.size})", ic.calculate_correct(test_X, test_y, weight_vec))

    x = [int(i) for i in range(w_cg.size)]
    
    plt.plot(x, w_cg , label = f"Conjugate GD ($\lambda = {reg}$)",
                marker = 'x', alpha = 0.8, linestyle = '--')
    plt.plot(x, w_qr, label = f"QR ($\lambda = {reg}$)",
                marker = 'o', alpha = 0.8, linestyle = '--')
    plt.plot(x, weight_vec, label = "Ordinary LSE",
                marker = '^', alpha = 0.8, linestyle = '--')
    plt.plot(x, w_bd, label = f"Bold Driver GD ($\lambda = {reg}$)",
                marker = '>', alpha = 0.8, linestyle = '--')

    plt.xlabel("Weight Vector Index", fontsize = 19)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.ylabel("Normalised Weight Value", fontsize=19)
    plt.title("Algorithm Comparision with %.1f fraction as training" % (frac), fontsize =22)
    plt.legend(frameon=True, fontsize=15)
    plt.tight_layout()
    plt.savefig("iris_figures/iris_gd_comparison", dpi = 300)
    


def main():
    # part_c()
    part_d_i()


if __name__ == "__main__":
    main()




