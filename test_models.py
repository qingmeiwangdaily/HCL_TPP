import torch
import TPPs.lrhp as lrhp
import TPPs.thp as thp

batch = 3
seq_len = 5
event_type = 4 * torch.rand(batch, seq_len)
event_type, _ = torch.sort(event_type, dim=1, descending=False)
event_type = event_type.type(torch.LongTensor)

time_type = torch.rand(batch, seq_len)
time_type, _ = torch.sort(time_type, dim=1, descending=False)
print(event_type, time_type)

num_type = 4
d_model = 2

thp_model = thp.TransformerHawkes(num_types=num_type, d_model=d_model)
enc_output2, enc_att2, all_lambda2, mask2, prediction2 = thp_model(event_type, time_type)

hp_model = lrhp.LowRankHawkes(num_types=num_type, d_model=d_model)
enc_output1, enc_att1, all_lambda1, mask1, prediction1 = hp_model(event_type, time_type)

print(enc_output1.shape, enc_output2.shape)
print(enc_att1.shape, enc_att2.shape)
print(all_lambda1.shape, all_lambda2.shape)
print(prediction1[0].shape, prediction2[0].shape)
print(prediction1[1].shape, prediction2[1].shape)
print(enc_att1)
print(enc_att2)
