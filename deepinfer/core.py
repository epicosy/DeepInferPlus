import keras
import numpy as np
import pandas as pd

from keras.src.engine.base_layer import Layer


PREDICTION_INTERVALS = [0.95]
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

    weights, biases, gammas, activation_functions, inverse_functions = zip(
        *[
            get_layer_representation(model.layers[i])
            for i in range(model_size)
        ]
    )

    return list(weights), list(biases), list(gammas), list(activation_functions), list(inverse_functions), model_size


def log(weights: list, biases: list, gammas: list, activation_functions: list):
    print(f"#weights: {len(weights)} | #biases: {len(biases)} | #N: {len(activation_functions)} | #Gamma: {len(gammas)}")

    for i in range(len(weights)):
        print("W_", i + 1, ":", weights[i])

    for i in range(len(biases)):
        print("B_", i + 1, ":", biases[i])

    for i in range(len(activation_functions)):
        print("Activation function of layer_a", i + 1, ":", activation_functions[i])

    for i in range(len(gammas)):
        print("Gamma_", i + 1, ":", gammas[i])


def infer_data_precondition(model: keras.Model, dataset: pd.DataFrame) -> dict:
    Q = []  # postconditions

    weights, biases, gammas, activation_functions, inverse_functions, model_size = get_abstract_representation(model)
    log(weights, biases, gammas, activation_functions)

    for i in range(model_size):
        for j in range(len(PREDICTION_INTERVALS)):
            for k in range(len(CONDITIONS)):
                M = np.matmul((inverse_functions[i] * PREDICTION_INTERVALS[j]), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[k], PREDICTION_INTERVALS[j])
                print(M)

    def beta(N, Q, l, i):
        """This is a recursive function
        to find the wp of N"""
        p = 0
        N0 = N[0]
        if l == 1:
            if N0 == 'linear':
                M = np.matmul((inverse_functions[i] * Q), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                print(M)

            if N0 == 'relu':
                try:
                    M1 = np.matmul((inverse_functions[i] * Q), - (biases[i]))
                except ValueError:
                    pass
                M2 = np.matmul(inverse_functions[i], - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                # print(M1)
                print(M2)
                M = M2

            if N0 == 'sigmoid':
                M = np.matmul((inverse_functions[i] * np.log(Q / (1 - Q))), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                print(M)

            if N0 == 'tanh':
                n_tanh = abs((Q - 1) / (Q + 1))
                M = np.matmul((inverse_functions[i] * (0.5 * np.log(n_tanh))), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                print(M)

            return M

        else:
            N0 = [N[0]]
            N1 = N[1:]
            # N0 = [N[:-1]]
            # N1 = N[-1:]
            l = len(N1)
            wp2 = beta(N1, Q, l, i)
            i = i - 1
            Q = wp2
            # N2 = N0
            # l = len(N2)
            # wp1 = beta(N0,wp2,1,i)
            # #wp = beta(N0,beta(N1,Q,l),1)
            # i = i - 1
            # wp0 = beta(N0,wp1,1,i)
            while (i >= 0):
                wp1 = beta(N0, Q, 1, i)
                i = i - 1
                Q = wp1
            return Q

    # TODO: check this; changed it to the first value in the precondition list, since the list is one element long
    Q = PREDICTION_INTERVALS[0]
    print(Q)
    # l=2
    i = model_size - 1

    wp = beta(activation_functions, Q, model_size, i)

    # TODO: check this if is correct, as it omits the last feature (original code does this too)
    features = dataset.columns.to_numpy()[:-1]

    print("wp: ")
    print(wp)
    print('Numpy Array: ', features)
    print('feature length: ', features.size)
    print('WP size ', wp.size)

    if wp.size == features.size:
        print("True")
        WPdictionary = dict(zip(features, wp))
        print(WPdictionary)
        for key, value in WPdictionary.items():
            print(key)
            # print(key, '>', "{0:.2f}".format(value))
    else:
        # print(WP_values)
        feature_counter = 1
        feature_count = np.array([])
        for i in wp:
            print("feature_counter", feature_counter, '>=', "{0:.2f}".format(i))
            feature_count = np.append(feature_count, feature_counter)
            feature_counter = feature_counter + 1
        WPdictionary = dict(zip(feature_count, wp))
        print(WPdictionary)

    for key, value in WPdictionary.items():
        print(key, '>=', value)

    return WPdictionary
