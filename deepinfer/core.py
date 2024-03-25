import keras
import numpy as np
import pandas as pd

from typing import Union, Tuple
from keras.src.engine.base_layer import Layer


PREDICTION_INTERVALS = [0.75, 0.90, 0.95, 0.99]
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
        # print("X_", i + 1, "postcondition:", CONDITIONS[0], post_condition)
        # print(M)

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


def infer_data_precondition(model: keras.Model, prediction_interval: float) -> np.ndarray:
    """
    This function computes the weakest precondition of a model given a test dataset.
    :param model: a Keras model
    :param prediction_interval: the prediction interval
    :return:
    """
    weights, biases, gammas, activation_functions, inverse_functions, n_layers = get_abstract_representation(model)
    log_representation(weights, biases, gammas, activation_functions)
    log_post_conditions(n_layers, inverse_functions, biases)

    # start with the last layer
    return compute_precondition(activation_functions, prediction_interval, n_layers=n_layers,
                                i=n_layers - 1, inverse_functions=inverse_functions, biases=biases)


def match_features_to_precondition(weakest_precondition: np.ndarray, dataset: pd.DataFrame) -> dict:
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
        weakest_precondition_dict = dict(zip(features[:-1], weakest_precondition))

        # feature_count = np.array([])
        #for i, wp in enumerate(weakest_precondition, start=1):
        #    print("feature_counter", i, condition, "{0:.2f}".format(wp))
        #    feature_count = np.append(feature_count, i)

        #weakest_precondition_dict = dict(zip(feature_count, weakest_precondition))

    return weakest_precondition_dict


# TODO: change hardcoded condition, the precondition should actually be a string of the type "binary_operator value"
def check_precondition_violation(x: pd.Series, condition: str, precondition: float) -> bool:
    return not pd.eval(f"{x} {condition} {precondition}")


def collect_feature_wise_violations(data: pd.DataFrame, weakest_preconditions: dict, condition: str) -> pd.DataFrame:
    """
    This function computes the weakest precondition violation for each feature in the dataset.
    :param data: a pandas Series with the features
    :param weakest_preconditions: a dictionary with the weakest precondition for each feature
    :param condition: the condition
    :return:
    """
    results = []

    for feature, precondition in weakest_preconditions.items():
        results.append(data[feature].apply(lambda x: check_precondition_violation(x, condition, precondition)))

    return pd.concat(results, axis=1)


def compute_threshold(model: keras.Model, dataset: pd.DataFrame, prediction_interval: float,
                      condition: str) -> Tuple[float, dict]:
    """
    This function computes the threshold for the weakest precondition violation.
    :param model: a Keras model
    :param dataset: the validation dataset to compute the threshold
    :param prediction_interval: the prediction interval
    :param condition: the condition
    :return:
    """
    wp = infer_data_precondition(model, prediction_interval)
    wp_dict = match_features_to_precondition(wp, dataset)

    violations = collect_feature_wise_violations(dataset, wp_dict, condition=condition)

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


def check_prediction(features: pd.DataFrame, threshold: float, wp_dict: dict, condition: str) \
        -> Tuple[pd.Series, pd.Series, pd.Series]:
    violations = collect_feature_wise_violations(features, wp_dict, condition)
    violations_count = violations[list(wp_dict)].eq(True).astype(int).sum()
    satisfaction_count = violations[list(wp_dict)].eq(False).astype(int).sum()

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

    return violations["implication"], violations_count, satisfaction_count
