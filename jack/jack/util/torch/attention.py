import torch
import torch.nn as nn
import torch.nn.functional as F

from jack.util.torch.esim_util import masked_softmax, weighted_sum

class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


class SelfAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                premise_dep_mask=None,
                visualize=False):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        _similarity_matrix = premise_batch.bmm(premise_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        if premise_dep_mask is not None:
            similarity_matrix = _similarity_matrix * premise_dep_mask
        else:
            similarity_matrix = _similarity_matrix

        prem_hyp_attn = masked_softmax(similarity_matrix, premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(premise_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        if visualize:
            if premise_dep_mask is not None:
                return attended_premises, _similarity_matrix, premise_dep_mask, prem_hyp_attn
            else:
                return attended_premises, _similarity_matrix, prem_hyp_attn
        else:
            return attended_premises


class DependencySelfAttention(nn.Module):
    
    def forward(self,
                premise_batch,
                premise_mask,
                dependency_mask):
        similarity_matrix = premise_batch.bmm(premise_batch.transpose(2, 1)
                                                    .contiguous())
        
        self_attn = masked_softmax(similarity_matrix, premise_mask)

        dep_attn = self_attn * dependency_mask

        attended_premises = weighted_sum(premise_batch,
                                         dep_attn,
                                         premise_mask)
        
        return attended_premises