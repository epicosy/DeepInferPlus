import keras
import numpy as np
import pandas as pd

PREDICTION_INTERVALS = [0.95]
CONDITIONS = ['>=', '<=', '>', '<', '==', '!=']


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
    num_model_layers = len(model.layers)
    Q = []  # postconditions

    print(f"Number of model's layers: {num_model_layers}")

    # Getting Weights and Biases for each layer and Compute Function variable \gamma
    weight_matrix = np.array([])
    weights = []  # Storing weight matrices
    biases = []  # Storing Biases
    gammas = []  # inverse functions (γ) of layer’s weight matrix
    activation_functions = []  # Activation function list
    Gamma_tr = []  # inverse functions (γ) of layer’s weight matrix
    X = []
    # if 'dense' in layer.name or 'Dense' in str(model.layers[i]):
    #    print("For dense based network")

    for i in range(0, num_model_layers):
        print(i)
        weight_matrix = model.layers[i].get_weights()[0]
        weights.append(weight_matrix)
        b = model.layers[i].get_weights()[1]
        biases.append(b)
        a = model.layers[i].get_config()['activation']
        activation_functions.append(a)
        w_tr = np.transpose(weight_matrix)
        # print(f'Array:\n{w}')
        # print(f'Transposed Array:\n{w_tr}')
        A = np.matmul(weight_matrix, w_tr)
        A_inv = np.linalg.inv(A)
        gamma = np.matmul(w_tr, A_inv)
        gammas.append(gamma)
        gamma_transpose = np.transpose(gamma)
        Gamma_tr.append(gamma_transpose)

    log(weights, biases, gammas, activation_functions)

    for i in range(num_model_layers):
        for j in range(len(PREDICTION_INTERVALS)):
            for k in range(len(CONDITIONS)):
                M = np.matmul((Gamma_tr[i] * PREDICTION_INTERVALS[j]), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[k], PREDICTION_INTERVALS[j])
                print(M)

    def beta(N, Q, l, i):
        """This is a recursive function
        to find the wp of N"""
        p = 0
        N0 = N[0]
        if l == 1:
            if N0 == 'linear':
                M = np.matmul((Gamma_tr[i] * Q), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                print(M)

            if N0 == 'relu':
                try:
                    M1 = np.matmul((Gamma_tr[i] * Q), - (biases[i]))
                except ValueError:
                    pass
                M2 = np.matmul(Gamma_tr[i], - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                # print(M1)
                print(M2)
                M = M2

            if N0 == 'sigmoid':
                M = np.matmul((Gamma_tr[i] * np.log(Q / (1 - Q))), - (biases[i]))
                print("X_", i + 1, "postcondition:", CONDITIONS[0], Q)
                print(M)

            if N0 == 'tanh':
                n_tanh = abs((Q - 1) / (Q + 1))
                M = np.matmul((Gamma_tr[i] * (0.5 * np.log(n_tanh))), - (biases[i]))
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
    i = num_model_layers - 1

    wp = beta(activation_functions, Q, num_model_layers, i)

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
