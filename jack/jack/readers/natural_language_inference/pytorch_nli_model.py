import torch
import torch.nn as nn
import torch.nn.functional as F

from jack.core import *
from jack.core.torch import PyTorchModelModule
from jack.util.torch.esim import ESIM, DepSAESIM

class PyTorchModularNLIModel(PyTorchModelModule):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config

        super(PyTorchModularNLIModel, self).__init__(shared_resources)
    
    @property
    def input_ports(self) -> List[TensorPort]:
        if self.shared_resources.embeddings is not None:
            if self.config.get("use_dep_sa", False):
                return [Ports.Input.emb_support, 
                        Ports.Input.support_length,
                        Ports.Input.support_dep_i,
                        Ports.Input.support_dep_j,
                        Ports.Input.support_dep_type,
                        Ports.Input.emb_question,
                        Ports.Input.question_length,
                        Ports.Input.question_dep_i,
                        Ports.Input.question_dep_j,
                        Ports.Input.question_dep_type,
                        Ports.is_eval]
            else:
                return [Ports.Input.emb_support, 
                        Ports.Input.support_length,
                        Ports.Input.emb_question,
                        Ports.Input.question_length,
                        Ports.is_eval]
        else:
            return [Ports.Input.support, 
                    Ports.Input.question,
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
        if shared_resources.config.get('use_dep_sa', False):
            prediction_module = DepSAESIM(
                embedding_dim=shared_resources.config.get('embedding_dim', 300),
            )
        else:
            prediction_module = ESIM(
                embedding_dim=shared_resources.config.get('embedding_dim', 300),
                self_attention=shared_resources.config.get('self_attention', False),
            )
        print('Prediction module:', prediction_module)
        return prediction_module
    
    def create_loss_module(self, shared_resources: SharedResources) -> nn.Module:
        loss = nn.CrossEntropyLoss()
        print('Loss module:', loss)
        return loss
