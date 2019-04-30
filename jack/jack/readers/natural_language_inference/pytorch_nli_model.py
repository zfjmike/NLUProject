import torch
import torch.nn as nn
import torch.nn.functional as F

from jack.core import *
from jack.core.torch import PyTorchModelModule
from jack.util.torch.esim import ESIM

class PyTorchModularNLIModel(PyTorchModelModule):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config

        super(PyTorchModularNLIModel, self).__init__(shared_resources)
    
    @property
    def input_ports(self) -> List[TensorPort]:
        if self.shared_resources.embeddings is not None:
            return [Ports.Input.emb_support, Ports.Input.emb_question,
                    # Ports.Input.support, Ports.Input.question,
                    Ports.Input.support_length, Ports.Input.question_length,
                    # character information
                    # Ports.Input.word_chars, Ports.Input.word_char_length,
                    # Ports.Input.question_batch_words, Ports.Input.support_batch_words,
                    Ports.is_eval]
        else:
            return [Ports.Input.support, Ports.Input.question,
                    Ports.Input.support_length, Ports.Input.question_length,
                    # character information
                    # Ports.Input.word_chars, Ports.Input.word_char_length,
                    # Ports.Input.question_batch_words, Ports.Input.support_batch_words, 
                    Ports.is_eval]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Prediction.candidate_index]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Target.target_index]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_prediction_module(self, shared_resources: SharedResources) -> nn.Module:
        prediction_module = ESIM()
        print('Prediction module:', prediction_module)
        return ESIM()
    
    def create_loss_module(self, shared_resources: SharedResources) -> nn.Module:
        loss = nn.CrossEntropyLoss()
        print('Loss module:', loss)
        return loss
