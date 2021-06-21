from Path import Path
from Layer import DenseLayer, DropOutLayer, ToDenseLayer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import List

"""
The data from our cooperation partner Econda used in the course of the analyses was provided to us on the condition that
it would not be forwarded and would only be used for research purposes.

Therefore, the following code section contains a fictitious use case regarding the recommender from 'SampleModel.png'.

It is shown how the stand-alone NN paths A_Path and B_Path can be reconstructed with the help of TransPer.
The same applies to the combined path as the overall network.
"""


class A_Path(Path):

    def __init__(self, initial_relevance, layer_names, activations, weights, bias, combined=False):

        Path.__init__(self, initial_relevance, layer_names)

        if not combined:

            layer_name = layer_names[-1]
            a_layer_4 = DenseLayer(1000, 50, activations[layer_name], weights[layer_name], bias[layer_name])
            self.layer_order.append(a_layer_4)

            layer_name = layer_names[-2]
            a_layer_3 = DropOutLayer(1000, 1000, activations[layer_name])
            self.layer_order.append(a_layer_3)

        layer_name = layer_names[-3]
        a_layer_2 = DenseLayer(500, 1000, activations[layer_name], weights[layer_name], bias[layer_name])
        self.layer_order.append(a_layer_2)

        layer_name = layer_names[-4]
        a_layer_1 = DenseLayer(50, 500, activations[layer_name], weights[layer_name], bias[layer_name])
        self.layer_order.append(a_layer_1)


class B_Path(Path):

    def __init__(self, initial_relevance, layer_names, activations, weights, bias, combined=False):

        Path.__init__(self, initial_relevance, layer_names)

        if not combined:

            layer_name = layer_names[-1]
            b_layer_4 = DenseLayer(100, 50, activations[layer_name], weights[layer_name], bias[layer_name])
            self.layer_order.append(b_layer_4)

        layer_name = layer_names[-3]
        b_layer_2 = DenseLayer(200, 100, activations[layer_name], weights[layer_name], bias[layer_name])
        self.layer_order.append(b_layer_2)

        layer_name = layer_names[-4]
        b_layer_1 = DenseLayer(300, 200, activations[layer_name], weights[layer_name], bias[layer_name])
        self.layer_order.append(b_layer_1)


class PathCombined(Path):

    def __init__(self, initial_relevance, layer_names, activations, weights, bias, combined=False):

        Path.__init__(self, initial_relevance, layer_names)

        a_path_order = layer_names['a_path']
        b_path_order = layer_names['b_path']
        combined_order = layer_names['combined']

        layer_name = combined_order[-1]

        comb_dense = DenseLayer(2000, 50, activations[layer_name], weights[layer_name], bias[layer_name])
        self.layer_order.append(comb_dense)

        activations_lower = activations['a_path'][-1].tolist() + activations['b_path'][-1].tolist()

        activations_lower = np.array(activations_lower)

        concatenate_layer = ToDenseLayer(2000, activations_lower)
        self.layer_order.append(concatenate_layer)

        comb_dense.lrp(concatenate_layer, self.initial_relevances)

        a_path_relevances = []
        b_path_relevances = []

        for i in range(2000):
            if i < 1000:
                a_path_relevances.append(comb_dense.relevances[i])
            else:
                b_path_relevances.append(comb_dense.relevances[i])

        self.a_path = A_Path(np.array(a_path_relevances), a_path_order, activations['a_path'], weights['a_path'],
                             bias['a_path'], combined)
        self.b_path = B_Path(np.array(b_path_relevances), b_path_order, activations['b_path'], weights['b_path'],
                             bias['b_path'], combined)

    def lrp(self):
        result = {}
        result['a_path'] = self.a_path.lrp()
        result['b_path'] = self.b_path.lrp()
        return result


def extract(model: Model, features: List[tf.Tensor]):
    """
    Extract the neuron activations, weights and bias for each layer
    :param model: Keras model used for prediction
    :param features: Input for a customer profile
    """

    # Do the prediction
    model_structure = model.layers[0]

    extractor = Model(inputs=model_structure.inputs,
                        outputs=[layer.output for layer in model_structure.layers])
    data = extractor(features)

    weights = {}
    bias = {}
    activations = {}

    for layer in model_structure.layers:

        weights[layer.name]= model_structure.get_layer('layer.name').weights[0].numpy()
        bias[layer.name] = model_structure.get_layer('layer.name').weights[1].numpy()

        i=0
        try:
            try:
                activations[layer.name] = data[i].numpy()
            except AttributeError:
                activations[layer.name] = None
        except TypeError:
            activations[layer.name] = None
        i += 1

    return activations, weights, bias


def __main__():

    a_path_order = ['a_input', 'a_hidden_layer_1', 'a_hidden_layer_2', 'a_output']
    b_path_order = ['b_input', 'b_hidden_layer_1', 'b_output']
    combined_order = ['concatenate']

    full_order = {'a_path': a_path_order, 'b_path': b_path_order, 'combined': combined_order}

    #########################################################################################
    #########################################################################################
    #########################################################################################

    """
    model) As mentioned above, we cannot provide specific customer data and trained models at this point.
    In our use case, models generated with Tensorflow and saved in HDF5 format were successfully used.
    (https://www.tensorflow.org/tutorials/keras/save_and_load)
    
    features) For each path, a specific input is provided in the form of a tf.Tensor.
    
    initial_relevance) The initial relevance is passed in the form of an np.array, 
    where its size corresponds to the size of the network's output layer.
    """
    model = None
    features = None
    initial_relevance = None
    activations, weights, bias = extract(model, features)

    #########################################################################################
    #########################################################################################
    #########################################################################################

    lrp_path = PathCombined(initial_relevance, full_order, activations, weights, bias, combined=True)
    explanation = lrp_path.lrp()
