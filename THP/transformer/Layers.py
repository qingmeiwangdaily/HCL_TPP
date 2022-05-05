import torch.nn as nn
import matplotlib.pyplot as plt

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import matplotlib.pyplot as plt

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask
        
        enc_slf_attn_map = enc_slf_attn.cpu().detach().numpy()
        plt.matshow(enc_slf_attn_map[0, 0, :, :])
        # plt.savefig("./1.png")
        plt.show()
        # plt.close()

        return enc_output, enc_slf_attn
