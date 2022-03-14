import torch.nn as nn
from .event_type import TypeEmbedding
from .position import PositionalEmbedding


class EventEmbedding(nn.Module):
    """
    Event Embedding which is consisted with under features
        1. TypeEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        3. Time: TBD
        sum of all these features are output of EventEmbedding
    """

    def __init__(self, type_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # self.type = TypeEmbedding(type_size=type_size, embed_size=embed_size)
        self.type = nn.Embedding(type_size+1, embed_size)
        self.position = PositionalEmbedding(d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.type(sequence) + self.position(sequence)
        return self.dropout(x)