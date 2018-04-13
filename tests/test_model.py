# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *

features = {"word": {"vocab_size": 100, "emb_size": 5, "dropout_p": 0.0}}
encoder = EncoderRNN(features=features, hidden_size=8, rnn_cell='lstm', n_layers=1, bidirectional=True)

print(encoder)

src_inputs = Variable(torch.LongTensor([[1, 2, 3], [3, 4, 0]]))
src_lengths = torch.LongTensor([3, 2])

encoder_outputs, encoder_hidden = encoder({'word': src_inputs}, src_lengths)
print(encoder_outputs, encoder_hidden)


decoder = DecoderRNN(100, 5, 8, 10, rnn_cell='lstm', use_attention=True, bidirectional=True)
print(decoder)
output_logits, _, res_dict = decoder.forward(src_inputs, encoder_hidden, encoder_outputs, src_lengths)

print(res_dict[DecoderRNN.KEY_SEQUENCE])
print(res_dict[DecoderRNN.KEY_ATTN_SCORE])
# for attn_dist in res_dict[DecoderRNN.KEY_ATTN_SCORE]:
#     print(attn_dist)
# for ids in res_dict[DecoderRNN.KEY_SEQUENCE]:
#     print(ids)


label_decoder = SeqLabelDecoderRNN(100, 5, 8, 10, rnn_cell='lstm', use_attention=True, bidirectional=True)
print(decoder)
label_init_inputs = Variable(torch.LongTensor([1, 1]))
output_logit2s, _, res_dict2 = label_decoder.forward(label_init_inputs, encoder_hidden, encoder_outputs, src_lengths)

print(res_dict2[DecoderRNN.KEY_SEQUENCE])
print(res_dict2[DecoderRNN.KEY_ATTN_SCORE])

pass
