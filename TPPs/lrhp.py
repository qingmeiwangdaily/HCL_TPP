import torch
import torch.nn as nn
from TPPs.utils import PAD, softplus, get_subsequent_mask, get_non_pad_mask, LinearPredictor


def d_time(time):
    """
    Input: batch * seq_len.
    Output: batch * seq_len * seq_len.
    """
    t_row = time.unsqueeze(2)
    t_col = time.unsqueeze(1)
    return t_row - t_col


class LowRankHawkes(nn.Module):
    """Classic Hawkes process model"""

    def __init__(self, num_types: int, d_model: int = 256, kernel: str = 'exp'):
        super(LowRankHawkes, self).__init__()
        self.num_types = num_types
        self.kernel = kernel
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)
        self.linear = nn.Linear(d_model, num_types + 1)
        self.base_intensity = nn.Parameter(torch.randn(1, 1, num_types))

        # prediction of next time stamp
        self.time_predictor = LinearPredictor(d_model, 1)
        # prediction of next event type
        self.type_predictor = LinearPredictor(d_model, num_types)

    def time_kernel(self, event_time, att_mask):
        """
        Input: batch * seq_len.
        Output: batch * seq_len * seq_len.
        """
        dt = d_time(event_time)
        return torch.exp(self.alpha * dt) * (1 - att_mask)

    def forward(self, event_type: torch.Tensor, event_time: torch.Tensor):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                enc_att: batch * seq_len * seq_len;
                all_lambda: batch * seq_len * data_type;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        non_pad_mask = get_non_pad_mask(event_type)  # batch * seq_len * 1
        slf_attn_mask_subseq = get_subsequent_mask(event_type)  # batch * seq_len * seq_len
        time_decay = self.time_kernel(event_time, slf_attn_mask_subseq)  # batch * seq_len * seq_len
        event_embeddings = non_pad_mask * self.event_emb(event_type)  # batch * seq_len * d_model

        mu = non_pad_mask * softplus(self.base_intensity, self.beta1)  # batch * seq_len * num_type
        infectivity = non_pad_mask * softplus(self.linear(event_embeddings), self.beta2)  # batch * seq_len * num_type+1
        all_lambda = mu + torch.bmm(time_decay, infectivity[:, :, 1:])  # batch * seq_len * num_type

        enc_att = torch.zeros_like(time_decay)
        for b in range(time_decay.shape[0]):
            events = event_type[b, :]
            enc_att[b, :, :] = infectivity[b, :, events] * time_decay[b, :, :]
        enc_output = torch.bmm(time_decay, event_embeddings)  # batch * seq_len * d_model
        time_prediction = self.time_predictor(enc_output, non_pad_mask)
        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, enc_att, all_lambda, non_pad_mask, (type_prediction, time_prediction)
