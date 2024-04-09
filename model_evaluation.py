import numpy as np
import scipy


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
