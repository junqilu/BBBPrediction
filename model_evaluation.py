import numpy as np
import scipy

def calc_f_stat(y_actual, y_modeled1, y_modeled2, parameter_model1,
               parameter_model2, num_data):
    """
    Calculates the f-statistic for comparing any models, including linear
    regression model, using the mean square error and mean square
    regression. Adopted from Module1 - Linear Regression & Fstat.ipynb on
    1.16.2024 lecture in-class code along

    Args:
        y_actual (np array): actual y values
        y_modeled1 (np_array): modeled y value from model1. If model1 is the
        mean model, y_modeled1 is
        np.mean(y)
        y_modeled2 (np_array): modeled y value from model2
        parameter_model1 (int): number of parameters for model1. If model1 is
        the mean model,
        p1 = 1 for mean
        parameter_model2 (int): number of parameters for model2. If model2 is
        the linear fit
        model, parameter_model2 = 2 for slope and intercept
        num_data (int): number of data.

    Returns:
        f_stat (float): calculated F-stat

    """

    sse_model1 = np.sum(
        (y_actual - y_modeled1) ** 2)  # model1 is the mean model
    sse_model2 = np.sum(
        (y_actual - y_modeled2) ** 2)  # model2 is the linear fit model

    f_stat = (sse_model1 - sse_model2) * (num_data - parameter_model2) / (
            sse_model2 * (parameter_model2 - parameter_model1))


    return f_stat


def calculate_structural_components(y_test, y_pred):
    #y_test and y_pred need to be np.array

    true_pos_indices = y_test == 1  #Get all the indices where y is 1
    true_neg_indices = y_test == 0  #Get all the indices where y is 0

    true_pos_count = sum(true_pos_indices)  #m
    true_neg_count = sum(true_neg_indices)  #n

    true_pos_probs = y_pred[true_pos_indices]  #X
    true_neg_probs = y_pred[true_neg_indices]  #Y
    # Example: if you have list = [1,2,3,4] and index = [0,1,0,1] (like for False and True), list[index] gives [2,4]

    # Calculate structural components
    V10 = []  #This is the V10 matrix
    for i in range(true_pos_count):
        V10_inner = []
        for j in range(true_neg_count):
            V10_inner.append(
                np.heaviside((true_pos_probs[i] - true_neg_probs[j]), 0.5))
        V10.append(np.mean(V10_inner))

    V01 = []  #For the V01, you just switch the outer loop in V10 caluclation
    # to the be inner loop
    for j in range(true_neg_count):
        V01_inner = []
        for i in range(true_pos_count):
            V01_inner.append(
                np.heaviside((true_pos_probs[i] - true_neg_probs[j]), 0.5))
        V01.append(np.mean(V01_inner))

    return V10, V01


def delongs_test(model_a_probs, model_b_probs, model_a_auc, model_b_auc,
                 y_test):
    true_pos_count = sum(y_test == 1)  #m
    true_neg_count = sum(y_test == 0)  #n

    #Calculating structural components
    model_a_V10, model_a_V01 = calculate_structural_components(
        y_test,
        model_a_probs
    )
    model_b_V10, model_b_V01 = calculate_structural_components(
        y_test,
        model_b_probs
    )
    print("Model A's V10: {}".format(model_a_V10))
    print("Model A's V01: {}".format(model_a_V01))
    print("Model B's V10: {}".format(model_b_V10))
    print("Model B's V01: {}".format(model_b_V01))

    #Calculate S matrices
    S10 = np.cov(model_a_V10, model_b_V10)
    S01 = np.cov(model_a_V01, model_b_V01)
    print('S10: {}'.format(S10))
    print('S01: {}'.format(S01))
    #Combiane S matrices
    S = (1 / true_pos_count) * S10 + (1 / true_neg_count) * S01
    print('S: {}'.format(S))

    #Calculate Z score
    det = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    Z = (model_a_auc - model_b_auc) / np.sqrt(det)
    print('Z score: {}'.format(Z))

    #Find the corresponding p-value
    p_value = scipy.stats.norm.sf(abs(Z)) * 2  #2-tailed
    print('p value: {}'.format(p_value))

    return p_value


