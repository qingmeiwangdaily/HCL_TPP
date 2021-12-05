import math

import torch
from torch import nn

from anhp.model.xfmr import EncoderLayer, MultiHeadAttention


class XFMRNHPFast(nn.Module):
    def __init__(self, dataset, d_model, n_layers, n_head, dropout, d_time, d_inner=128, use_norm=False,
                 sharing_param_layer=False):
        # d_inner only used if we want to add feedforward
        super(XFMRNHPFast, self).__init__()
        self.d_model = d_model
        self.d_time = d_time

        self.div_term = torch.exp(torch.arange(0, d_time, 2) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        # here num_types already includes [PAD], [BOS], [EOS]
        self.Emb = nn.Embedding(dataset.num_types, d_model, padding_idx=dataset.pad_index)
        self.n_layers = n_layers
        self.n_head = n_head
        self.sharing_param_layer = sharing_param_layer
        if not sharing_param_layer:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [EncoderLayer(
                            d_model + d_time,
                            MultiHeadAttention(1, d_model + d_time, d_model, dropout, output_linear=False),
                            # PositionwiseFeedForward(d_model + d_time, d_inner, dropout),
                            use_residual=False,
                            dropout=dropout
                        )
                            for _ in range(n_layers)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        else:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [EncoderLayer(
                            d_model + d_time,
                            MultiHeadAttention(1, d_model + d_time, d_model, dropout, output_linear=False),
                            # PositionwiseFeedForward(d_model + d_time, d_inner, dropout),
                            use_residual=False,
                            dropout=dropout
                        )
                            for _ in range(0)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.inten_linear = nn.Linear(d_model * n_head, dataset.event_num)
        self.softplus = nn.Softplus()
        self.eps = torch.finfo(torch.float32).eps
        # self.add_bos = dataset.add_bos
        self.add_bos = True

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe

    def forward_pass(self, init_cur_layer_, tem_enc, tem_enc_layer, enc_input, combined_mask, batch_non_pad_mask=None):
        cur_layers = []
        seq_len = enc_input.size(1)
        for head_i in range(self.n_head):
            cur_layer_ = init_cur_layer_
            for layer_i in range(self.n_layers):
                layer_ = torch.cat([cur_layer_, tem_enc_layer], dim=-1)
                _combined_input = torch.cat([enc_input, layer_], dim=1)
                if self.sharing_param_layer:
                    enc_layer = self.heads[head_i][0]
                else:
                    enc_layer = self.heads[head_i][layer_i]
                enc_output = enc_layer(
                    _combined_input,
                    combined_mask
                )
                if batch_non_pad_mask is not None:
                    _cur_layer_ = enc_output[:, seq_len:, :] * (batch_non_pad_mask.unsqueeze(-1))
                else:
                    _cur_layer_ = enc_output[:, seq_len:, :]

                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_
                enc_input = torch.cat([enc_output[:, :seq_len, :], tem_enc], dim=-1)
                # non-residual connection
                # cur_layer_ = torch.tanh(_cur_layer_)

                # enc_output *= _combined_non_pad_mask.unsqueeze(-1)
                # layer_ = torch.tanh(enc_output[:, enc_input.size(1):, :])
                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def forward(self, event_seqs, time_seqs, batch_non_pad_mask, attention_mask, extra_times=None):
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = torch.tanh(self.Emb(event_seqs))
        init_cur_layer_ = torch.zeros_like(enc_input)
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(
            attention_mask.device)
        if extra_times is None:
            tem_enc_layer = tem_enc
        else:
            tem_enc_layer = self.compute_temporal_embedding(extra_times)
            tem_enc_layer *= batch_non_pad_mask.unsqueeze(-1)
        # batch_size * (seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # batch_size * (2 * seq_len) * (2 * seq_len)
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
        _combined_mask = torch.cat([contextual_mask, _combined_mask], dim=1)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_enc_layer, enc_input, _combined_mask, batch_non_pad_mask)

        return cur_layer_

    def compute_loglik(self, batch):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = batch
        # 1. compute event-loglik
        enc_out = self.forward(event_seq[:, :-1], time_seq[:, :-1], batch_non_pad_mask[:, 1:], attention_mask[:, 1:, :-1], time_seq[:, 1:])
        enc_inten = self.softplus(self.inten_linear(enc_out))
        # original: 1->1, 2->2
        # event_lambdas = torch.sum(enc_inten * type_mask, dim=2) + self.eps
        # now: 1->2, 2->3
        event_lambdas = torch.sum(enc_inten * type_mask[:, 1:], dim=2) + self.eps
        # in case event_lambdas == 0
        # event_lambdas.masked_fill_(~batch_non_pad_mask, 1.0)
        event_lambdas.masked_fill_(~batch_non_pad_mask[:, 1:], 1.0)

        event_ll = torch.log(event_lambdas)
        res_enc_inten = enc_inten

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        num_samples = 100
        # 2.1 sample times
        # 2.2 compute intensities at sampled times
        # due to GPU memory limitation, we may not be able to compute all intensities at all sampled times,
        # step gives the batch size w.r.t how many sampled times we should process at each batch
        step = 20
        if not self.add_bos:
            extended_time_seq = torch.cat([torch.zeros(time_seq.size(0), 1).to(time_seq.device), time_seq], dim=-1)
            diff_time = (time_seq[:, :] - extended_time_seq[:, :-1]) * batch_non_pad_mask[:, :]
            temp_time = diff_time.unsqueeze(0) * \
                        torch.rand([num_samples, *diff_time.size()], device=time_seq.device)
            temp_time += extended_time_seq[:, :-1].unsqueeze(0)
            all_lambda = self._compute_intensities_fast(event_seq, time_seq, batch_non_pad_mask, attention_mask,
                                                        temp_time, step)
        else:
            # why non_pad_mask start from 1?
            # think about a simple case: [e] [e] [pad] (non_pad_mask: 1 1 0)
            # you want to compute the first interval only, so if you use non_pad_mask[:, :-1] (1, 1),
            # you will compute both the first and the second intervals!
            diff_time = (time_seq[:, 1:] - time_seq[:, :-1]) * batch_non_pad_mask[:, 1:]
            temp_time = diff_time.unsqueeze(0) * \
                        torch.rand([num_samples, *diff_time.size()], device=time_seq.device)
            temp_time += time_seq[:, :-1].unsqueeze(0)
            # for interval computation, we will never use the last event -- that is why we have -1 in
            # event_seq, time_seq, attention_mask
            all_lambda = self._compute_intensities_fast(event_seq[:, :-1], time_seq[:, :-1], batch_non_pad_mask[:, 1:],
                                                        attention_mask[:, 1:, :-1],
                                                        temp_time, step)

        # sum over type_events, then sum over sampled times
        all_lambda = all_lambda.sum(dim=-1)

        # 2.3 compute the empirical expectation of the summation
        all_lambda = all_lambda.sum(dim=0) / num_samples
        non_event_ll = all_lambda * diff_time

        # return enc_inten to compute accuracy
        return event_ll, non_event_ll, res_enc_inten

    def _compute_intensities_fast(self, event_seq, time_seq, batch_non_pad_mask, attention_mask, temp_time, step=20):
        # fast version, can only use in log-likelihood computation
        # assume we will sample the same number of times in each interval of the event_seqs
        all_lambda = []
        batch_size = event_seq.size(0)
        seq_len = event_seq.size(1)
        num_samples = temp_time.size(0)
        for i in range(0, num_samples, step):
            _extra_time = temp_time[i: i + step, :, :]
            _step = _extra_time.size(0)
            _extra_time = _extra_time.reshape(_step * batch_size, -1)
            _types = event_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _times = time_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _batch_non_pad_mask = batch_non_pad_mask.unsqueeze(0).expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _attn_mask = attention_mask.unsqueeze(0).expand(_step, -1, -1, -1).reshape(_step * batch_size, seq_len,
                                                                                       seq_len)
            _enc_output = self.forward(_types, _times,
                                       _batch_non_pad_mask,
                                       _attn_mask,
                                       _extra_time)
            all_lambda.append(self.softplus(self.inten_linear(_enc_output)).reshape(_step, batch_size, seq_len, -1))
        all_lambda = torch.cat(all_lambda, dim=0)
        return all_lambda

    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        # Assumption: all the sampled times are distributed [time_seq[...,-1], next_event_time]
        # used for thinning algorithm
        num_batches = event_seq.size(0)
        seq_len = event_seq.size(1)
        assert num_batches == 1, "Currently, no support for batch mode (what is a good way to do batching in thinning?)"
        if num_batches == 1 and num_batches < sampled_times.size(0):
            _sample_size = sampled_times.size(0)
            # multiple sampled_times
            event_seq = event_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            time_seq = time_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            num_batches = event_seq.size(0)
        assert (time_seq[:, -1:] <= sampled_times).all(), "sampled times must occur not earlier than last events!"
        num_samples = sampled_times.size(1)

        # 1. prepare input embeddings for "history"
        tem_enc = self.compute_temporal_embedding(time_seq)
        enc_input = torch.tanh(self.Emb(event_seq))
        init_cur_layer_ = torch.zeros((sampled_times.size(0), sampled_times.size(1), enc_input.size(-1))).to(
            sampled_times.device)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        tem_layer_ = self.compute_temporal_embedding(sampled_times)

        # 2. prepare attention mask
        attention_mask = torch.ones((num_batches, seq_len + num_samples, seq_len + num_samples)).to(event_seq.device)
        attention_mask[:, :seq_len, :seq_len] = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0).cuda()
        # by default, regard all_sampled times to be equal to the last_event_time
        # recall that we use 1 for "not attending", 0 for "attending"
        # t_i == sampled_t
        attention_mask[:, seq_len:, :seq_len - 1] = 0
        # t_i < sampled_t
        attention_mask[:, seq_len:, seq_len - 1][time_seq[:, -1:] < sampled_times] = 0
        attention_mask[:, seq_len:, seq_len:] = (torch.eye(num_samples) < 1).unsqueeze(0).to(event_seq.device)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_layer_, enc_input, attention_mask)

        sampled_intensities = self.softplus(self.inten_linear(cur_layer_))

        return sampled_intensities
