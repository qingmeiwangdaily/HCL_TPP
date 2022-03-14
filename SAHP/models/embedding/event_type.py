import torch.nn as nn


class TypeEmbedding(nn.Embedding):
    def __init__(self, type_size, embed_size, padding_idx):
        super().__init__(type_size, embed_size, padding_idx=padding_idx)# padding_idx not 0