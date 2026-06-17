"""VMax (voltage maxima) training — pure Python, no compiled dependency.

Mirrors the pybind11 VMax training API from EventProp.Python.
Also defines ``OneLayerVMax`` which was missing from the original submodule.
"""

from abc import abstractmethod
from typing import NamedTuple, Tuple

from .training import AbstractTraining, AbstractTwoLayer
from .lif_layer import LIFLayer, LIFLayerParameters
from .loss_layer import VMaxCrossEntropyLoss, VMaxCrossEntropyLossParameters


class OneLayerVMax(AbstractTraining):
    """Single LIF layer followed directly by VMax cross-entropy loss.

    Architecture: input -> output_layer (LIF) -> loss (VMaxCrossEntropyLoss).
    """

    @abstractmethod
    def __init__(
        self,
        output_parameters: LIFLayerParameters,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.output_parameters = output_parameters
        self.output_layer = LIFLayer(self.output_parameters)

    def forward(self, minibatch):
        self.loss(self.output_layer(minibatch.spikes))

    def backward(self, minibatch):
        self.loss.backward(minibatch.labels)

    def get_weight_copy(self) -> Tuple:
        return (self.output_layer.w_in.copy(), self.loss.w_in.copy())


class TwoLayerVMax(AbstractTwoLayer):
    """Two LIF layers followed by VMax cross-entropy loss.

    Architecture: input -> hidden_layer (LIF) -> loss (VMaxCrossEntropyLoss).
    """

    def __init__(
        self,
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(),
        **kwargs,
    ):
        super().__init__(
            hidden_layer_class=LIFLayer,
            hidden_parameters=hidden_parameters,
            loss_class=VMaxCrossEntropyLoss,
            loss_parameters=loss_parameters,
            **kwargs,
        )
