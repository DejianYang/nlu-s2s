# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable


embs = nn.Embedding(10, 3)
rnn = nn.LSTM(3, 2, bidirectional=True)

input_vars = Variable(torch.LongTensor([[1, 3, 2], [3, 4, 0]]))


input_lengths = torch.LongTensor([3, 2])

emb_inputs = embs.forward(input_vars)
if input_lengths is not None:
    emb_inputs = nn.utils.rnn.pack_padded_sequence(emb_inputs, input_lengths.tolist(), batch_first=True)
outputs, hidden = rnn.forward(emb_inputs)
if input_lengths is not None:
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

print(outputs[:, -1, :])
print(hidden)

logits = Variable(torch.rand(2, 3))
predict_ids = torch.topk(logits, dim=1, k=1)[1]
print(logits)
print(predict_ids)

