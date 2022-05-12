import torch
from TPPs.utils import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask


def d_time(time, non_pad_mask):
    """
    Input: batch * seq_len.
    Output: batch * seq_len * seq_len.
    """
    t_row = time.unsqueeze(2)
    t_col = time.unsqueeze(1)
    return (t_row - t_col) * non_pad_mask


batch = 3
seq_len = 5
event_type = 5 * torch.rand(batch, seq_len)
event_type, _ = torch.sort(event_type, dim=1, descending=False)
event_type = event_type.type(torch.LongTensor)

time_type = torch.rand(batch, seq_len)
time_type, _ = torch.sort(time_type, dim=1, descending=False)
print(event_type, time_type)

non_pad_mask = get_non_pad_mask(event_type)
print('Non pad mask:', non_pad_mask.size())
print(non_pad_mask)

slf_attn_mask_subseq = get_subsequent_mask(event_type)
print('Attn_mask mask:', slf_attn_mask_subseq.size())
print(slf_attn_mask_subseq)

dt = d_time(time_type, 1 - slf_attn_mask_subseq)
print('dt')
print(dt)

slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
print('Key pad:', slf_attn_mask_keypad.size())
print(slf_attn_mask_keypad)

slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
print('Att mask:', slf_attn_mask.size())
print(slf_attn_mask)
