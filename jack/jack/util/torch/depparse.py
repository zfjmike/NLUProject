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
        
        self.dep_embedding = nn.Embedding(num_dep_types, 1)
        self.pipeline = stanfordnlp.Pipeline(tokenize_pretokenized=True, use_gpu=use_gpu)
        self.type2id = self.pipeline.processors['depparse'].trainer.vocab['deprel']
        self.use_gpu = use_gpu
        if use_gpu:
            self.dep_embedding.cuda()
    
    def forward(self, token_lists, seq_len):
        batch_size = len(token_lists)
        dep_mask = torch.ones(batch_size, seq_len, seq_len)
        if self.use_gpu:
            dep_mask = dep_mask.cuda()

        sent = preprocess_tokens(token_lists)
        res = self.pipeline(sent)

        for i, sent in enumerate(res.sentences):
            for j, dep in enumerate(sent.dependencies):
                idx1 = dep[0].index
                idx2 = dep[2].index
                emb_idx = torch.Tensor(self.type2id(sent._dependencies[1]))
                if self.use_gpu:
                    emb_idx = emb_idx.cuda()
                emb = self.dep_embedding(emb_idx)
                dep_mask[i][idx1][idx2] = emb

        return dep_mask