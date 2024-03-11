import keras
import numpy as np
import pandas as pd

from typing import Union, Tuple
from keras.src.engine.base_layer import Layer

PREDICTION_INTERVALS = [0.95]
# Binary comparison operators
CONDITIONS = ['>=', '<=', '>', '<', '==', '!=']


def get_layer_representation(layer: Layer):
    """
    This function computes the inverse function (γ) of a layer’s weight matrix (W) where γ (W) ::= (Wt.W) −1
    :param layer: a layer of a model
    :return: tuple: inverse function, gamma, weight matrix, bias, activation function
    """
    weight_matrix = layer.get_weights()[0]
    bias = layer.get_weights()[1]
    activation_func = layer.get_config()['activation']
    weight_matrix_tr = np.transpose(weight_matrix)
    symmetric_matrix = np.matmul(weight_matrix, weight_matrix_tr)
    symmetric_matrix_inverse = np.linalg.inv(symmetric_matrix)
    gamma = np.matmul(weight_matrix_tr, symmetric_matrix_inverse)
    gamma_transpose = np.transpose(gamma)

    return gamma_transpose, gamma, weight_matrix, bias, activation_func


def get_abstract_representation(model: keras.Model):
    """
    This function computes the abstract representation of a model.

    :param model: A Keras model.
    :return:
        - weights (List[np.ndarray]): List of weight matrices for each layer.
        - biases (List[np.ndarray]): List of bias vectors for each layer.
        - gammas (List[np.ndarray]): List of gamma matrices for each layer.
        - activation_functions (List[str]): List of activation functions for each layer.
        - inverse_functions (List[np.ndarray]): List of inverse gamma matrices for each layer.
    """
    model_size = len(model.layers)

    print(f"Number of layers in the model: {model_size}")

    inverse_functions, gammas, weights, biases, activation_functions = zip(
        *[
            get_layer_representation(model.layers[i])
            for i in range(model_size)
        ]
    )

    return list(weights), list(biases), list(gammas), list(activation_functions), list(inverse_functions), model_size


def log_representation(weights: list, biases: list, gammas: list, activation_functions: list):
    print(
        f"#weights: {len(weights)} | #biases: {len(biases)} | #N: {len(activation_functions)} | #Gamma: {len(gammas)}")

    for i in range(len(weights)):
        print("W_", i + 1, ":", weights[i])

    for i in range(len(biases)):
        print("B_", i + 1, ":", biases[i])

    for i in range(len(activation_functions)):
        print("Activation function of layer_a", i + 1, ":", activation_functions[i])

    for i in range(len(gammas)):
        print("Gamma_", i + 1, ":", gammas[i])


# TODO: check name of the function and add docstrings
def log_post_conditions(model_size: int, inverse_functions: list, biases: list):
    for i in range(model_size):
        for j in range(len(PREDICTION_INTERVALS)):
            for k in range(len(CONDITIONS)):
                # TODO: find what the variable M means
                M = np.matmul((inverse_functions[i] * PREDICTION_INTERVALS[j]), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[k], PREDICTION_INTERVALS[j])
                print(M)


def compute_precondition(activation_functions: list, post_condition: Union[float, np.array], n_layers: int, i: int,
                         inverse_functions: list, biases: list) -> np.ndarray:
    """This is a recursive function to find the wp of N
    :param activation_functions: list of activation functions for each layer.
    :param post_condition: the post_condition of the model
    :param n_layers: number of layers
    :param i: index of the layer
    :param inverse_functions: list of inverse gamma matrices for each layer.
    :param biases: list of bias vectors for each layer.
    :return: (np.ndarray): precondition
    """

    if n_layers == 1:
        if activation_functions[0] == 'linear':
            M = np.matmul((inverse_functions[i] * post_condition), - (biases[i]))

        if activation_functions[0] == 'relu':
            try:
                # TODO: check if this is dead code
                M1 = np.matmul((inverse_functions[i] * post_condition), - (biases[i]))
            except ValueError:
                pass
            # TODO: why is this not using the value of M1?
            M = np.matmul(inverse_functions[i], - (biases[i]))

        if activation_functions[0] == 'sigmoid':
            M = np.matmul((inverse_functions[i] * np.log(post_condition / (1 - post_condition))), - (biases[i]))

        if activation_functions[0] == 'tanh':
            n_tanh = abs((post_condition - 1) / (post_condition + 1))
            M = np.matmul((inverse_functions[i] * (0.5 * np.log(n_tanh))), - (biases[i]))

        # TODO: add cases for other activation functions
        print("X_", i + 1, "postcondition:", CONDITIONS[0], post_condition)
        print(M)

        return M

    else:
        # pass the activation function from all layers except for the first
        wp2 = compute_precondition(activation_functions[1:], post_condition, len(activation_functions) - 1, i,
                                   inverse_functions, biases)
        i = i - 1
        post_condition = wp2

        while i >= 0:
            # pass the activation function from the first layer
            wp1 = compute_precondition([activation_functions[0]], post_condition, 1, i,
                                       inverse_functions, biases)
            i = i - 1
            # TODO: check if this is correct as previously it was overwriting variable from outer scope (Q)
            post_condition = wp1
        return post_condition


def infer_data_precondition(model: keras.Model) -> np.ndarray:
    """
    This function computes the weakest precondition of a model given a test dataset.
    :param model: a Keras model
    :return:
    """
    weights, biases, gammas, activation_functions, inverse_functions, n_layers = get_abstract_representation(model)
    log_representation(weights, biases, gammas, activation_functions)
    log_post_conditions(n_layers, inverse_functions, biases)

    # TODO: check this; changed it to the first value in the PREDICTION_INTERVALS, since the list is one element long
    post_conditions = PREDICTION_INTERVALS[0]
    print(post_conditions)

    # start with the last layer
    return compute_precondition(activation_functions, post_conditions, n_layers=n_layers,
                                i=n_layers - 1, inverse_functions=inverse_functions, biases=biases)


def match_features_to_precondition(weakest_precondition: np.ndarray, dataset: pd.DataFrame) -> dict:
    condition = CONDITIONS[0]
    # TODO: check this if is correct, as it omits the last feature (original code does this too)
    # Is correct for Unseen set but incorrect for Test set, since it leaves out the last
    # feature (which is the label for unseen)
    features = dataset.columns.to_numpy()

    print(f"Weakest precondition: {weakest_precondition}")
    print('Features: ', features)
    print('Features length: ', features.size)
    print('Weakest precondition size ', weakest_precondition.size)

    if weakest_precondition.size == features.size:
        # TODO: what does this mean?
        weakest_precondition_dict = dict(zip(features, weakest_precondition))

        for key, value in weakest_precondition_dict.items():
            print(key)
    else:
        feature_count = np.array([])
        weakest_precondition_dict = dict(zip(features[:-1], weakest_precondition))

        #for i, wp in enumerate(weakest_precondition, start=1):
        #    print("feature_counter", i, condition, "{0:.2f}".format(wp))
        #    feature_count = np.append(feature_count, i)

        #weakest_precondition_dict = dict(zip(feature_count, weakest_precondition))

    for key, value in weakest_precondition_dict.items():
        print(key, condition, value)

    return weakest_precondition_dict


# TODO: change hardcoded condition, the precondition should actually be a string of the type "binary_operator value"
def check_precondition_violation(x: pd.Series, precondition: float) -> bool:
    return not pd.eval(f"{x} {CONDITIONS[0]} {precondition}")


def collect_feature_wise_violations(data: pd.DataFrame, weakest_preconditions: dict) -> pd.DataFrame:
    """
    This function computes the weakest precondition violation for each feature in the dataset.
    :param data: a pandas Series with the features
    :param weakest_preconditions: a dictionary with the weakest precondition for each feature
    :return:
    """
    results = []

    for feature, precondition in weakest_preconditions.items():
        print(data.columns)
        results.append(data[feature].apply(lambda x: check_precondition_violation(x, precondition)))

    return pd.concat(results, axis=1)


def compute_threshold(model: keras.Model, dataset: pd.DataFrame) -> Tuple[float, dict]:
    """
    This function computes the threshold for the weakest precondition violation.
    :param model: a Keras model
    :param dataset: the validation dataset to compute the threshold
    :return:
    """
    wp = infer_data_precondition(model)
    wp_dict = match_features_to_precondition(wp, dataset)

    violations = collect_feature_wise_violations(dataset, wp_dict)

    return violations[list(wp_dict)].eq(True).astype(int).sum(axis='columns').mean(), wp_dict


def decision_tree(x: pd.Series, threshold: float):
    if x['more_imp_feat_viol_counter'] == 0:
        return 'Correct'

    if x['less_imp_feat_viol_counter'] > threshold > x['more_imp_feat_viol_counter']:
        return 'Wrong'

    if x['less_imp_feat_viol_counter'] != x['more_imp_feat_viol_counter']:
        return 'Wrong'

    if x['less_imp_feat_viol_counter'] == x['more_imp_feat_viol_counter'] == threshold:
        return 'Uncertain'

    if x['less_imp_feat_viol_counter'] < threshold < x['more_imp_feat_viol_counter']:
        return 'Correct'

    # TODO: add edge case where less_imp_feat_viol_counter < threshold and threshold > more_imp_feat_viol_counter


def check_prediction(model: keras.Model, features: pd.DataFrame, labels: pd.DataFrame, threshold: float, wp_dict: dict,
                     invert: bool = False) -> dict:
    violations = collect_feature_wise_violations(features, wp_dict)
    violations_count = violations[list(wp_dict)].eq(True).astype(int).sum()

    important_features = [str(i) for i, v in violations_count.items() if v <= violations_count.mean()]
    unimportant_features = [str(i) for i, v in violations_count.items() if v > violations_count.mean()]

    violations['more_imp_feat_viol_counter'] = violations[important_features].eq(True).astype(int).sum(axis='columns')
    violations['less_imp_feat_viol_counter'] = violations[unimportant_features].eq(True).astype(int).sum(axis='columns')

    # TODO: the implication should be computed by the decision tree function
    #violations["implication"] = violations.apply(lambda x: decision_tree(x, threshold), axis=1)

    violations["implication"] = violations.apply(
        lambda x: "Correct" if (x['more_imp_feat_viol_counter'] == 0) else "Uncertain",
        axis=1)
    violations["implication"] = violations.apply(lambda x: "Wrong" if (x['less_imp_feat_viol_counter'] > threshold and
                                                                       x['more_imp_feat_viol_counter'] < threshold and
                                                                       x['less_imp_feat_viol_counter'] != x[
                                                                           'more_imp_feat_viol_counter']) else "Correct",
                                                 axis=1)
    violations["implication"] = violations.apply(
        lambda x: "Uncertain" if (x['less_imp_feat_viol_counter'] == x['more_imp_feat_viol_counter'] and
                                  x['more_imp_feat_viol_counter'] != 0) else x["implication"],
        axis=1)

    violations['Predicted_Outcome'] = (model.predict(features) > 0.5).astype(int)
    violations['Actual_Outcome'] = labels

    violations["GroundTruth"] = violations.apply(
        lambda x: "Correct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "Wrong", axis=1)

    if invert:
        violations["TrueNegative"] = violations.apply(
            lambda x: "TN" if (x['GroundTruth'] == 'Correct' and x['implication'] == 'Correct') else "Not", axis=1)

        violations["FalseNegative"] = violations.apply(
            lambda x: "FN" if (x['implication'] == 'Correct' and x['GroundTruth'] == 'Wrong') else "Not", axis=1)

        violations["TruePositive"] = violations.apply(
            lambda x: "TP" if (x['GroundTruth'] == 'Wrong' and x['implication'] == 'Wrong') else "Not", axis=1)

        violations["FalsePositive"] = violations.apply(
            lambda x: "FP" if (x['GroundTruth'] == 'Correct' and x['implication'] == 'Wrong') else "Not", axis=1)

    else:
        violations["TruePositive"] = violations.apply(
            lambda x: "TP" if (x['GroundTruth'] == 'Correct' and x['implication'] == 'Correct') else "Not", axis=1)

        violations["FalsePositive"] = violations.apply(
            lambda x: "FP" if (x['implication'] == 'Correct' and x['GroundTruth'] == 'Wrong') else "Not", axis=1)

        violations["TrueNegative"] = violations.apply(
            lambda x: "TN" if (x['GroundTruth'] == 'Wrong' and x['implication'] == 'Wrong') else "Not", axis=1)

        violations["FalseNegative"] = violations.apply(
            lambda x: "FN" if (x['GroundTruth'] == 'Correct' and x['implication'] == 'Wrong') else "Not", axis=1)

    violations["ActualFalsePositive"] = violations.apply(
        lambda x: "TPAct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "FPAct", axis=1)

    results = {
        'GT_Correct': violations["GroundTruth"].str.contains('Correct', regex=False).sum().astype(int),
        'GT_Wrong': violations["GroundTruth"].str.contains('Wrong', regex=False).sum().astype(int),
    }

    results.update({
        '#Correct': violations["implication"].str.contains('Correct', regex=False).sum().astype(int),
        '#Wrong': violations["implication"].str.contains('Wrong', regex=False).sum().astype(int),
        '#Uncertain': violations["implication"].str.contains('Uncertain', regex=False).sum().astype(int)
    })

    results.update({
        'TP': violations["TruePositive"].str.contains('TP', regex=False).sum().astype(int),
        'FP': violations["FalsePositive"].str.contains('FP', regex=False).sum().astype(int),
        'TN': violations["TrueNegative"].str.contains('TN', regex=False).sum().astype(int),
        'FN': violations["FalseNegative"].str.contains('FN', regex=False).sum().astype(int)
    })

    fpr = results['FP'] / (results['FP'] + results['TN']) if results['FP'] + results['TN'] != 0 else 0
    tpr = results['TP'] / (results['TP'] + results['FN']) if results['TP'] + results['FN'] != 0 else 0

    results.update({'FPR': round(fpr * 100, 2), 'TPR': round(tpr * 100, 2)})

    precision = results['TP'] / (results['TP'] + results['FP']) if results['TP'] + results['FP'] != 0 else 0
    recall = results['TP'] / (results['TP'] + results['FN']) if results['TP'] + results['FN'] != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    results.update({'Precision': round(precision * 100, 2), 'Recall': round(recall * 100, 2), 'F1': round(f1 * 100, 2)})

    mcc = (results['TP'] * results['TN'] - results['FP'] * results['FN']) / np.sqrt(
        (results['TP'] + results['FP']) * (results['TP'] + results['FN']) * (results['TN'] + results['FP']) * (
                results['TN'] + results['FN']))

    results.update({'MCC': round(mcc, 3)})

    results.update({
        'ActFP': violations["ActualFalsePositive"].str.contains('FPAct', regex=False).sum().astype(int),
        'ActTP': violations["ActualFalsePositive"].str.contains('TPAct', regex=False).sum().astype(int),
        'Acc': len(violations[violations["Predicted_Outcome"] == violations['Actual_Outcome']]) / len(violations)
    })

    results.update({'#Violation': violations_count.sum()})
    results.update({'#Satisfaction': violations[list(wp_dict)].eq(False).astype(int).sum().sum()})

    return results
