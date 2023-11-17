import iris_classification as ic
import matplotlib.pyplot as plt
import numpy as np

iris_data_loc = './data/iris.csv' # File name of IRIS dataset
data_array = ic.extract_dataset(iris_data_loc)



def part_c(frac = 0, weight_samples = 10000):

    """ Sahir explanation for having samples generated which give better weights than
    the least-squares solution is the additon of noise to the weights can also act
    as means of regularisation, such as by reducing overfititng or the effects of outliers
    in some instances. It can be difficult to predict where to add the noise however, i.e.
    in which weights"""

    # generate data and outcome
    train_X, train_y, test_X, test_y = ic.split_data_rnd(data_array, fraction=frac)

    n_corr, w_ref = ic.cluster_dataset_test(test_X,test_y,test_X,test_y)

    w_vars = w_ref + (np.random.uniform(size = (weight_samples,test_X.shape[1]))-0.5)
    
    corr_higher = 0

    for i in w_vars:

        if n_corr < ic.calculate_correct(test_X,test_y,i):

            corr_higher +=1


    print(f"In {weight_samples} random weight samples, {corr_higher} higher than least-squares solution")




part_c()




