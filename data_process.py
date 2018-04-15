# -*- coding:utf-8 -*-
import os
import random


def load_session_data(raw_data_path):
    samples = []
    count = 0
    with open(raw_data_path, 'r', encoding='utf-8') as fr:
        session_sample = []
        for line in fr:
            line = line.strip()
            if len(line) <= 0:
                if len(session_sample) > 0:
                    samples += [session_sample]
                    session_sample = []
                continue
            ss = line.split('\t')
            assert len(ss) == 5
            session_sample += [ss]
            count += 1
    lengths = [len(s) for s in samples]
    assert sum(lengths) == count
    return samples
    pass


def assert_sample(sample):
    assert len(sample) == 5
    assert len(sample[0].split()) == len(sample[1].split()), sample
    assert len(sample[0].split()) == len(sample[2].split()), sample
    assert len(sample[0].split()) == len(sample[3].split()), sample
    assert 1 == len(sample[4].split())
    pass


def save_samples(samples, saved_path):
    count = 0
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for sample in samples:
            for session_sample in sample:
                count += 1
                assert_sample(session_sample)
                fw.write('\t'.join(session_sample)+'\n')
    print('...save {} samples into {}'.format(count, saved_path))
    pass


def split_dataset(raw_data_path, saved_dir, valid_rate=0.1):
    samples = load_session_data(raw_data_path)
    print('...load {} session samples from {}'.format(len(samples), raw_data_path))

    valid_num = int(len(samples) * valid_rate)
    ids = list(range(len(samples)))
    random.shuffle(ids)
    train_samples = [samples[i] for i in ids[:-valid_num]]
    valid_samples = [samples[i] for i in ids[-valid_num:]]
    print(valid_num, len(train_samples), len(valid_samples))
    save_samples(train_samples, os.path.join(saved_dir, 'train.tsv'))
    save_samples(valid_samples, os.path.join(saved_dir, 'valid.tsv'))
    pass


def load_data(path):
    return [s for s in open(path, 'r', encoding='utf-8')]


if __name__ == '__main__':
    split_dataset('./data/corpus.txt', './data/')
    pass
