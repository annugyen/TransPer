import numpy as np
import math

def get_relevance(activations, weights, bias, relevances_in, lrp_algorithm, params):

    if lrp_algorithm == 'standard':
        return get_relevance_standard(activations, weights, bias, relevances_in)
    elif lrp_algorithm == 'epsilon':
        return get_relevance_epsilon(activations, weights, bias, relevances_in, params[0])
    elif lrp_algorithm == 'alphabeta':
        return get_relevance_alphabeta(activations, weights, bias, relevances_in, params[0])
    elif lrp_algorithm == 'nonnegative':
        return get_relevance_nonnegative(activations, weights, bias, relevances_in)
    elif lrp_algorithm == 'gamma':
        return get_relevance_gamma(activations, weights, bias, relevances_in, params[0])
    else:
        return get_relevance_standard(activations, weights, bias, relevances_in)

def get_relevance_standard(activations, weights, bias, relevances_in):
    """
    Simple version of the LRP algorithm (see LRP-algorithm_1.png)
    :param activations: activations of lower layer (input of this layer)
    :param weights: weights of this layer
    :param bias: bias of this layer
    :param relevances_in: relevances of upper layer
    """

    norm_sums = activations.dot(weights) + bias

    rel_div_sums = np.divide(relevances_in, norm_sums)

    pre_result = weights.dot(rel_div_sums)

    result = np.multiply(activations, pre_result)

    return result

def get_relevance_epsilon(activations, weights, bias, relevances_in, epsilon=0.01):
    """
    Alternate version of the LRP algorithm with additional parameter epsilon to gain better numerical properties.
    The obtained explanations are typically less noisy and consist of less input features.
    :param activations: activations of lower layer (input of this layer)
    :param weights: weights of this layer
    :param bias: bias of this layer
    :param relevances_in: relevances of upper layer
    :param epsilon: stabilizer
    :return: relevances_out: relevances of the layer
    """

    norm_sums = activations.dot(weights) + bias

    norm_sums_e = norm_sums+epsilon*np.sign(norm_sums)

    rel_div_sums = np.divide(relevances_in, norm_sums_e)

    pre_result = weights.dot(rel_div_sums)

    relevances_out = np.multiply(activations, pre_result)

    return relevances_out

def get_relevance_alphabeta(activations, weights, bias, relevances_in, alpha=1):
    """
    Alternate version of the LRP algorithm with additional parameters alpha and beta to treat positive and negative
    components differently.
    :param activations: activations of lower layer (input of this layer)
    :param weights: weights of this layer
    :param bias: bias of this layer
    :param relevances_in: relevances of upper layer
    :param alpha: factor for positive z
    :param beta: factor for negative z
    :return: relevances_out: relevances of the layer
    """

    beta = alpha - 1

    activated_weights = (weights.T * activations).T

    activated_weights_p = activated_weights.copy()
    activated_weights_n = activated_weights.copy()

    activated_weights_p[activated_weights_p < 0] = 0
    activated_weights_n[activated_weights_n > 0] = 0

    bias_p = bias.copy()
    bias_n = bias.copy()
    bias_p[bias_p < 0] = 0
    bias_n[bias_n > 0] = 0

    norm_sums_p = activated_weights_p.sum(axis=0) + bias_p
    norm_sums_n = activated_weights_n.sum(axis=0) + bias_n

    rel_div_sums_p = np.divide(relevances_in, norm_sums_p, out=np.zeros_like(relevances_in), where=norm_sums_p!=0)
    rel_div_sums_n = np.divide(relevances_in, norm_sums_n, out=np.zeros_like(relevances_in), where=norm_sums_n!=0)

    pre_result_p = alpha * activated_weights_p.dot(rel_div_sums_p)
    pre_result_n = beta * activated_weights_n.dot(rel_div_sums_n)

    relevances_out = pre_result_p + pre_result_n

    return relevances_out

def get_relevance_nonnegative(activations, weights, bias, relevances_in):
    """
    Alternate version of the LRP algorithm that only treats positive weights.
    :param activations: activations of lower layer (input of this layer)
    :param weights: weights of this layer
    :param bias: bias of this layer
    :param relevances_in: relevances of upper layer
    :return: relevances_out: relevances of the layer
    """

    weights_p = weights.copy()

    weights_p[weights_p < 0] = 0

    norm_sums_p = activations.dot(weights_p) + bias

    rel_div_sums = np.divide(relevances_in, norm_sums_p)

    pre_result = weights.dot(rel_div_sums)

    relevances_out = np.multiply(activations, pre_result)

    return relevances_out

def get_relevance_gamma(activations, weights, bias, relevances_in,gamma=0.5):
    """
    Alternate version of the LRP algorithm that gives more weight to positive weights.
    :param activations: activations of lower layer (input of this layer)
    :param weights: weights of this layer
    :param bias: bias of this layer
    :param relevances_in: relevances of upper layer
    :param gamma: (1+gamma) is factor for positive weights
    :return:
    """

    weights_p = weights.copy()

    weights_p[weights_p < 0] = 0

    weights_new = weights+gamma*weights_p

    norm_sums_p = activations.dot(weights_new) + (1+gamma) * bias

    rel_div_sums = np.divide(relevances_in, norm_sums_p)

    pre_result = weights_new.dot(rel_div_sums)

    relevances_out = np.multiply(activations, pre_result)

    return relevances_out

def get_relevance_gru(layer_lower, indices_relevances_upper, weights, return_seq, lrp_algorithm, params):
    """
    Determines the relevance for a GRU layer w.r.t the 'return_seq' parameter
    """

    len_seq = len(layer_lower.neuron_activations)
    neuron_act = layer_lower.neuron_activations

    kernel_weights = weights[0].np()
    recurrent_weights = weights[1].np()
    bias = weights[2].np()

    kernel_weights_fixed = {}
    recurrent_weights_fixed = {}
    bias_fixed = {'kernel': {}, 'recurrent': {}}

    length = int(len(kernel_weights[0])/3)
    kernel_weights_fixed['update'] = kernel_weights[ : , 0:length]
    kernel_weights_fixed['reset'] = kernel_weights[ : , length:2*length]
    kernel_weights_fixed['current'] = kernel_weights[ : , 2*length:3*length]

    length = int(len(recurrent_weights[0]) / 3)
    recurrent_weights_fixed['update'] = recurrent_weights[ : , 0:length]
    recurrent_weights_fixed['reset'] = recurrent_weights[ : , length:2*length]
    recurrent_weights_fixed['current'] = recurrent_weights[ : , 2*length:3*length]

    y = np.array_split(bias[0], 3)
    bias_fixed['kernel']['update'] = y[0]
    bias_fixed['kernel']['reset'] = y[1]
    bias_fixed['kernel']['current'] = y[2]

    y = np.array_split(bias[1], 3)
    bias_fixed['recurrent']['update'] = y[0]
    bias_fixed['recurrent']['reset'] = y[1]
    bias_fixed['recurrent']['current'] = y[2]

    if return_seq:
        r_hidden_state = indices_relevances_upper[len_seq-1]
    else:
        r_hidden_state = indices_relevances_upper

    hidden_state_size = len(r_hidden_state)

    lrp_info = []
    hidden_state = np.zeros(hidden_state_size)

    for input in neuron_act:
        info = gru_forward(input, kernel_weights_fixed, recurrent_weights_fixed, bias_fixed, hidden_state_size,
                           hidden_state_size, hidden_state)
        hidden_state = info[0]
        lrp_info.append(info)

    r_cw_input = []



    for k in reversed(range(len_seq)):

        neuron_act_x = neuron_act[k]

        # f -> d,e

        activations = np.append(lrp_info[k][1], lrp_info[k][2])

        weights = []

        for i in range(len(activations)):
            weights.append([0] * (int(len(activations)/2)))
            if i < hidden_state_size:
                weights[i][i] = 1
            else:
                weights[i][i-hidden_state_size] = 1

        weights = np.array(weights)

        bias_temp = [0] * hidden_state_size

        bias_temp = np.array(bias_temp)

        r = np.split(get_relevance(activations, weights, bias_temp, r_hidden_state, lrp_algorithm, params), 2)

        r_e = r[0]
        r_d = r[1]

        # e -> hidden state

        r_hidden_state = r_e

        # d -> b

        r_b = r_d

        # b -> a, current_k

        activations = np.append(lrp_info[k][5], lrp_info[k][6])

        weights = []

        for i in range(len(activations)):
            weights.append([0] * (int(len(activations) / 2)))
            if i < hidden_state_size:
                weights[i][i] = 1
            else:
                weights[i][i - hidden_state_size] = 1

        weights = np.array(weights)

        r = np.split(get_relevance(activations, weights, bias_temp, r_b, lrp_algorithm, params), 2)

        r_a = r[0]
        r_current_k = r[1]

        # a -> current_r

        r_current_r = r_a

        # r_current_k -> r_current_k_input

        r_current_k_input = get_relevance(neuron_act_x, kernel_weights_fixed['current'], bias_fixed['kernel']['current'], r_current_k, lrp_algorithm, params)

        r_cw_input.append(r_current_k_input)

        # r_current_r -> r_hs

        r_hs = get_relevance(lrp_info[k - 1][0], recurrent_weights_fixed['current'], bias_fixed['recurrent']['current'], r_current_r, lrp_algorithm, params)

        # r_hs, r_hidden_state -> r_hs_input

        if k >= 1:
            if return_seq:
                r_hidden_state = r_hs + r_hidden_state + indices_relevances_upper[k-1]
            else:
                r_hidden_state = r_hs + r_hidden_state

    r_cw_input.reverse()

    return r_cw_input

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    #return np.maximum(0, np.minimum(1, (x + 2) / 4))

def oneminus(x):
    return 1 - x

def forward_pass_fc(input, weights):
    return input.dot(weights)

def gru_forward(input, kernel_weights_fixed, recurrent_weights_fixed, bias_fixed, lower_shape, upper_shape, hidden_state):
    """
    Simulates a forward pass of a GRU cell and stores the activations that occur in it
    :param input: Input of GRU cell
    :param kernel_weights_fixed: Processed kernel weights (see get_relevances_gru)
    :param recurrent_weights_fixed: Processed recurrent weights (see get_relevances_gru)
    :param bias_fixed: Processed bias (see get_relevances_gru)
    :param lower_shape: input dim
    :param upper_shape: output dim
    :param hidden_state: initial hidden state
    :return:
    """

    sigmoid_v = np.vectorize(sigmoid)
    oneminus_v = np.vectorize(oneminus)

    # UPDATE

    update_k = forward_pass_fc(input, kernel_weights_fixed['update']) + bias_fixed['kernel']['update']
    update_r = forward_pass_fc(hidden_state, recurrent_weights_fixed['update']) + bias_fixed['recurrent']['update']

    update_kr = update_k + update_r
    update_kr = sigmoid_v(update_kr)

    # RESET

    reset_k = forward_pass_fc(input, kernel_weights_fixed['reset']) + bias_fixed['kernel']['reset']
    reset_r = forward_pass_fc(hidden_state, recurrent_weights_fixed['reset']) + bias_fixed['recurrent']['reset']

    reset_kr = reset_k + reset_r
    reset_kr = sigmoid_v(reset_kr)

    # CURRENT

    current_k = forward_pass_fc(input, kernel_weights_fixed['current']) + bias_fixed['kernel']['current']
    current_r = forward_pass_fc(hidden_state, recurrent_weights_fixed['current']) + bias_fixed['recurrent']['current']

    # FURTHER COMPUTATION

    a = np.multiply(current_r, reset_kr)

    b = a + current_k

    b = np.tanh(b)

    c = oneminus_v(update_kr)

    d = np.multiply(b, c)

    e = np.multiply(update_kr, hidden_state)

    f = d + e

    return f, e, d, c, b, a, current_k, current_r