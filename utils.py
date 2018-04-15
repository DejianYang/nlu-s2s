# -*- coding:utf-8 -*-
import torch


def _clean(words):
    return list(filter(lambda word: word != '<sos>' and word != '<eos>'and word != '<pad>'and word != '<unk>', words))


def seq2words(vocab, seq):
    words = [vocab.itos[w] for w in seq]
    words = _clean(words)
    return words


def evaluate_slot_filling(vocab, target_seqs, predict_seqs, lengths):
    target_words = [seq2words(vocab, seq[:length]) for seq, length in zip(target_seqs, lengths)]
    predict_words = [seq2words(vocab, seq[:length]) for seq, length  in zip(predict_seqs, lengths)]
    # print('target', target_words)
    # print('predict', predict_words)

    exact_match = 0
    sent_num = 0
    for target_sent, predict_sent in zip(target_words, predict_words):
        # if len(target_sent) != len(predict_sent):
        #     print('predict size do not match, target: {}, predict: {}'.format(
        #         ' '.join(target_sent), ' '.join(predict_sent)))
        sent_num += 1
        if ' '.join(target_sent) == ' '.join(predict_sent):
            exact_match += 1
    # print(exact_match, sent_num)
    return exact_match, sent_num
