import keras
import numpy as np
import pandas as pd

from typing import Union
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
    print(f"#weights: {len(weights)} | #biases: {len(biases)} | #N: {len(activation_functions)} | #Gamma: {len(gammas)}")

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
    # TODO: check this if is correct, as it omits the last feature (original code does this too)
    features = dataset.columns.to_numpy()[:-1]

    print(f"Weakest precondition: {weakest_precondition}")
    print('Features: ', features)
    print('Features length: ', features.size)
    print('Weakest precondition size ', weakest_precondition.size)

    if weakest_precondition.size == features.size:
        # TODO: what does this mean?
        print("True")
        weakest_precondition_dict = dict(zip(features, weakest_precondition))

        for key, value in weakest_precondition_dict.items():
            print(key)
    else:
        feature_count = np.array([])

        for i, wp in enumerate(weakest_precondition, start=1):
            print("feature_counter", i, '>=', "{0:.2f}".format(wp))
            feature_count = np.append(feature_count, i)

        weakest_precondition_dict = dict(zip(feature_count, weakest_precondition))

    for key, value in weakest_precondition_dict.items():
        print(key, '>=', value)

    return weakest_precondition_dict
