import stanfordnlp

import torch
import torch.nn as nn
import torch.nn.functional as F

num_dep_types = 53

def preprocess_tokens(token_lists):
    ret = ''
    for i, token_list in enumerate(token_lists):
        for j, token in enumerate(token_list):
            ret = ret + (' ' if j > 0 else '') + token
        if i < len(token_lists) - 1:
            ret += '\n'
    return ret


class DependencyGenerator(nn.Module):

    def __init__(self, use_gpu=True):
        super(DependencyGenerator, self).__init__()

        self.dep_embedding = nn.Embedding(num_dep_types, 1, padding_idx=0)
        self.use_gpu = use_gpu
        if use_gpu:
            self.dep_embedding.cuda()
    
    def forward(self, dep_i, dep_j, dep_type, seq_len):
        """
        dep_i: [b * seq_len-1]
        dep_j: [b * seq_len-1]
        dep_type: [b * seq_len-1]
        seq_len: int

        return: [b * seq_len * seq_len]
        """
        batch_size = dep_i.size(0)
        dep_mask = torch.ones(batch_size, seq_len * seq_len)
        if self.use_gpu:
            dep_mask = dep_mask.cuda()
        
        dep_idx = dep_i * seq_len + dep_j
        dep_k = self.dep_embedding(dep_type).view(batch_size, -1)
        dep_mask.scatter_(1, dep_idx, dep_k)

        dep_mask = dep_mask.view(batch_size, seq_len, seq_len)

        return dep_mask