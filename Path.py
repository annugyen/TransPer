from abc import ABC

class Path(ABC):
    """
    Abstract class for Paths used in NN
    """

    def __init__(self, initial_relevance, layer_names, combined=False):
        """
        This abstract class is used to reconstruct the paths used in the NN model.
        :param initial_relevance: relevance of the output layer:
        For the desired index the NN value is used for this index. All other relevancies are set to 0.
        :param combined: true, if combined path is used
        """

        self.initial_relevance = initial_relevance
        self.layer_names = layer_names
        self.combined = combined
        self.layer_order = []

    def lrp(self):
        """
        runs through all layers defined for this path and calculates the relevance one after the other
        """

        length = len(self.layer_order)

        for i in range(length):

            if i != 0 and i != length - 1:
                self.layer_order[i].lrp(self.layer_order[i + 1], self.layer_order[i - 1].relevance)
            if i == 0:
                self.layer_order[i].lrp(self.layer_order[i + 1], self.initial_relevance)

        return self.layer_order[length - 2].relevance


