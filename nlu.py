# -*- coding:utf-8 -*
import os
import argparse
import logging
import json
import collections

import torch
from torch import optim
from models import *
from data_loader import *
from trainer import *

LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)


def load_config(config_path):
    model_config = json.load(open(config_path, 'r', encoding='utf-8'))
    logging.info('----------------model config-----------------------')
    for key in model_config:
        logging.info('{}: {}'.format(key, model_config[key]))
    logging.info('---------------------------------------------------')

    return model_config


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.tsv')
    valid_path = os.path.join(data_dir, 'valid.tsv')

    train_set, valid_set, fields_dict = get_dataset(train_path=train_path, valid_path=valid_path)

    logging.info('load train {} samples from {}'.format(len(list(train_set.examples)), train_path))
    logging.info('load valid {} samples from {}'.format(len(list(valid_set.examples)), valid_path))
    return train_set, valid_set, fields_dict


def build_model(model_config, fields_dict):
    input_feature_options = collections.OrderedDict()
    # add word config
    vocab1 = fields_dict[SRC_WORD_FIELD].vocab

    input_feature_options[SRC_WORD_FIELD] = {"emb_size": model_config['features'][SRC_WORD_FIELD]['emb_size'],
                                             "dropout_p": model_config['features'][SRC_WORD_FIELD]['dropout_p'],
                                             "vocab_size": len(vocab1),
                                             "pad_idx": vocab1.stoi['<pad>']}

    # add tag1 config
    vocab2 = fields_dict[SRC_TAG1_FIELD].vocab
    input_feature_options[SRC_TAG1_FIELD] = {"emb_size": model_config['features'][SRC_TAG1_FIELD]['emb_size'],
                                             "dropout_p": model_config['features'][SRC_TAG1_FIELD]['dropout_p'],
                                             "vocab_size": len(vocab2),
                                             "pad_idx": vocab2.stoi['<pad>']}

    # add tag2 config
    vocab3 = fields_dict[SRC_TAG2_FIELD].vocab
    input_feature_options[SRC_TAG2_FIELD] = {"emb_size": model_config['features'][SRC_TAG2_FIELD]['emb_size'],
                                             "dropout_p": model_config['features'][SRC_TAG2_FIELD]['dropout_p'],
                                             "vocab_size": len(vocab3),
                                             "pad_idx": vocab3.stoi['<pad>']}

    for field_name in input_feature_options:
        logging.info('filed name: {}, config: {}'.format(field_name, input_feature_options[field_name]))

    slot_vocab = fields_dict[TGT_SLOT_FIELD].vocab
    intent_vocab = fields_dict[TGT_INTENT_FIELD].vocab

    encoder = EncoderRNN(features=input_feature_options,
                         rnn_cell=model_config['rnn_cell'],
                         n_layers=model_config['n_layers'],
                         hidden_size=model_config['hidden_size'],
                         bidirectional=model_config['bidirectional'],
                         dropout_p=model_config['dropout_p'])

    decoder = SeqLabelDecoderRNN(vocab_size=len(slot_vocab),
                                 emb_size=model_config['features'][TGT_SLOT_FIELD]['emb_size'],
                                 hidden_size=model_config['hidden_size'],
                                 max_length=None,
                                 input_dropout_p=model_config['features'][TGT_SLOT_FIELD]['dropout_p'],
                                 rnn_cell=model_config['rnn_cell'],
                                 n_layers=model_config['n_layers'],
                                 bidirectional=model_config['bidirectional'],
                                 dropout_p=model_config['dropout_p'],
                                 use_attention=True)
    classifier = nn.Linear(model_config['hidden_size'], len(intent_vocab))

    model = SlotLabelSeq2Seq(encoder, decoder, classifier)

    for param in model.parameters():
        param.data.uniform_(-model_config['init_w'], model_config['init_w'])

    logging.info('building model over ...')
    logging.info(model)

    optimizer = Optimizer(optimizer=optim.Adam(params=model.parameters(), lr=model_config['lr']),
                          max_grad_norm=model_config['max_grad_norm'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, 1000, 0.8)
    optimizer.set_scheduler(scheduler)

    logging.info(optimizer)
    logging.info('loss mask idx {}'.format(slot_vocab.stoi['<pad>']))
    loss = SeqLabelLoss(len(slot_vocab), slot_vocab.stoi['<pad>'], label_size=len(intent_vocab))
    logging.info(loss)

    return model, optimizer, loss
    pass


def main(args):
    model_config = load_config(args.config)
    train_set, valid_set, fields_dict = load_data(model_config['data_dir'])

    model, optimizer, loss = build_model(model_config, fields_dict)

    trainer = Trainer(loss,
                      expt_dir=model_config['expt_dir'],
                      batch_size=model_config['batch_size'],
                      display_freq=model_config['display_freq'],
                      ckpt_freq=model_config['ckpt_freq'])
    trainer.train(model=model, optimizer=optimizer,
                  num_epochs=model_config['num_epochs'],
                  train_set=train_set, valid_set=valid_set)
    pass


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Intent Detection and Slot Filling with SEQ2SEQ')

    # data
    parser.add_argument("--config", type=str, default="./data/config.json", help="config path")
    parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
    pass
