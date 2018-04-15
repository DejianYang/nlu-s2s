# -*- coding:utf-8 -*-
import torch
from data_loader import *

train_set, valid_set, fields_dict = get_dataset('../data/train.tsv', '../data/valid.tsv')

for key in fields_dict:
    print('field {}, vocab size: {}'.format(key, len(fields_dict[key].vocab)))


print(len(train_set.fields[SRC_TAG1_FIELD].vocab))

device = None if torch.cuda.is_available() else -1
valid_iter = torchtext.data.BucketIterator(
    dataset=valid_set, batch_size=5,
    sort=False, sort_key=lambda x: len(getattr(x, SRC_WORD_FIELD)),
    device=device, train=False)

for batch_input in valid_iter.__iter__():
    print(getattr(batch_input, SRC_WORD_FIELD))

    print(getattr(batch_input, SRC_TAG1_FIELD).size())
    print(getattr(batch_input, TGT_SLOT_FIELD).size())
    print(getattr(batch_input, TGT_SLOT_FIELD))
    print(getattr(batch_input, TGT_INTENT_FIELD).size())
    break

