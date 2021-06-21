from abc import ABC, abstractmethod
from LrpMethods import get_relevance, get_relevance_gru


class Layer(ABC):
    """
    This abstract class is used to reconstruct the layers used in the NN model.
    """

    def __init__(self, lower_shape, upper_shape, neuron_activations, dropout=False, weights=None, bias = None,
                 recurrent = False, return_seq = False):
        """
        Initialize a certain Layer
        :param lower_shape: number of input neurons, int
        :param upper_shape: number of output neurons, int
        :param neuron_activations: neuron activations of this layer, np.ndarray
        :param dropout: true, if droput layer, bool
        :param weights: weights of this layer (non-existent connections are represented by 0-connections), np.ndarray
        :param bias: bias of this layer (non-existent bias is represented by 0-values), np.ndarray
        :param recurrent: True if recurrent layer, bool
        :param return_seq: only if recurrent is True: True, if all hidden states of recurrent cells are output, bool
        """
        self.lower_shape = lower_shape
        self.upper_shape = upper_shape
        self.neuron_activations = neuron_activations
        self.dropout = dropout
        if not self.dropout:
            self.weights = weights
            self.bias = bias
            self.recurrent = recurrent
            if self.recurrent:
                self.return_seq = return_seq
        self.relevance = {}

    @abstractmethod
    def lrp(self, lower_layer, upper_relevance):
        """
        :param lower_layer: The lower (input) layer
        :param upper_relevance: already calculated relevance of the upper (outpu) layer
        """
        pass


class DropOutLayer(Layer):
    """
    This class is used to reconstruct dropout layers that don't modify the activations.
    """

    def __init__(self, lower_shape, upper_shape, neuron_activations):
        Layer.__init__(self, lower_shape, upper_shape, neuron_activations, dropout=True, recurrent=False)

    def lrp(self, lower_layer, upper_relevance, lrp_algorithm=None, params=None):
        self.relevance = upper_relevance


class DenseLayer(Layer):
    """
    This class is used to reconstruct dense layers.
    """

    def __init__(self, lower_shape, upper_shape, neuron_activations, weights, bias):
        Layer.__init__(self, lower_shape, upper_shape, neuron_activations, dropout=False, weights=weights, bias=bias,
                 recurrent=False)

    def lrp(self, lower_layer, upper_relevance, lrp_algorithm=None, params=None):
        self.relevance = get_relevance(lower_layer.neuron_activations, self.weights, self.bias, upper_relevance, lrp_algorithm, params)


class GRULayer(Layer):
    """
    This class is used to reconstruct GRU layers.
    The 'return_sequences' variant from https://keras.io/api/layers/recurrent_layers/gru/
    is also available via the 'return_seq' parameter.
    """
    def __init__(self, lower_shape, upper_shape, neuron_activations, weights, return_seq):
        Layer.__init__(self, lower_shape, upper_shape, neuron_activations, dropout=False, weights=weights,
                       recurrent=True, return_seq=return_seq)

    def lrp(self, lower_layer, upper_relevance, lrp_algorithm=None, params=None):
        self.relevance = get_relevance_gru(lower_layer, upper_relevance, self.weights, self.return_seq, lrp_algorithm, params)


class ToDenseLayer(Layer):
    """
    This class is used to reconstruct lrp behaviour within the concatenation layers.
    """
    def __init__(self, upper_shape, neuron_activations):
        Layer.__init__(self, None, upper_shape, neuron_activations)

    def lrp(self, lower_layer, upper_relevance, lrp_algorithm, params=None):
        pass


