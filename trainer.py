# -*- coding:utf-8 -*-

import logging
import os
import math
import random
import time
import itertools

import torch
import torch.nn as nn
from data_loader import *
from utils import *


class SeqLabelLoss(object):
    def __init__(self, seq_size, mask_idx, label_size, alpha=1.0):
        seq_mask = torch.ones(seq_size)
        seq_mask[mask_idx] = 0.0
        label_mask = torch.ones(label_size)

        self.seq_criterion = nn.NLLLoss(weight=seq_mask)
        self.label_criterion = nn.CrossEntropyLoss(weight=label_mask)

        self.seq_loss = 0.0
        self.label_loss = 0.0

        self.seq_norm = 0
        self.label_norm = 0

        self.alpha = alpha
        self.mask_idx = mask_idx
        pass

    def cuda(self):
        self.seq_criterion.cuda()
        self.label_criterion.cuda()

    def reset(self):
        self.seq_loss = 0.0
        self.label_loss = 0.0
        self.seq_norm = 0
        self.label_norm = 0
        pass

    def eval_step_seq(self, seq_logits, seq_target):
        self.seq_loss += self.seq_criterion(seq_logits, seq_target)
        self.seq_norm += seq_target.data.ne(self.mask_idx).sum()
        pass

    def eval_label(self, label_logits, label_target):
        self.label_loss += self.label_criterion(label_logits, label_target)
        self.label_norm += 1

    def backward(self):
        total_loss = self.seq_loss + self.alpha * self.label_loss
        total_loss.backward()
        pass

    def get_loss(self):
        seq_loss = self.seq_loss.data[0]
        nll = seq_loss / self.seq_norm
        ppl = math.exp(nll)

        label_loss = self.label_loss.data[0]
        label_loss /= self.label_norm

        total_loss = (self.seq_loss + self.alpha * self.label_loss).data[0]
        total_loss /= self.label_norm
        return total_loss, ppl, label_loss


class Optimizer(object):
    """ The Optimizer class encapsulates torch.optim package and provides functionalities
    for learning rate scheduling and gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.SGD(params)
        max_grad_norm (float, optional): value used for gradient norm clipping,
            set 0 to disable (default 0)
    """

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optimizer, max_grad_norm=0):
        self.optimizer = optimizer
        self.scheduler = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        """ Set the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler.*): object of learning rate scheduler,
               e.g. torch.optim.lr_scheduler.StepLR
        """
        self.scheduler = scheduler

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary. """
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        """ Update the learning rate if the criteria of the scheduler are met.

        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
            epoch (int): The current epoch number.
        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()


class Trainer(object):
    def __init__(self, loss: SeqLabelLoss, expt_dir='expt',
                 batch_size=32, display_freq=100, ckpt_freq=1000):

        self.loss = loss
        if torch.cuda.is_available():
            self.loss.cuda()

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        self.batch_size = batch_size
        self.display_freq = display_freq
        self.ckpt_freq = ckpt_freq

        # set when training
        self.model = None
        self.optimizer = None
        pass

    def _train_batch(self, batch_data):
        # get fields data
        word_inputs, input_lengths = getattr(batch_data, SRC_WORD_FIELD)
        tag1_inputs = getattr(batch_data, SRC_TAG1_FIELD)
        tag2_inputs = getattr(batch_data, SRC_TAG2_FIELD)
        seq_outputs = getattr(batch_data, TGT_SLOT_FIELD)
        label_outputs = getattr(batch_data, TGT_INTENT_FIELD)

        # print(word_inputs.size(), tag1_inputs.size(), tag2_inputs.size(), seq_outputs.size(), label_outputs.size())

        src_feat_inputs = {SRC_WORD_FIELD: word_inputs,
                           SRC_TAG1_FIELD: tag1_inputs,
                           SRC_TAG2_FIELD: tag2_inputs}

        tgt_var_inputs = seq_outputs[:, 0]
        seq_output_targets = seq_outputs[:, 1:]

        # Forward propagation
        label_output_logits, seq_output_logits, _ = self.model(src_feat_inputs, input_lengths, tgt_var_inputs)

        # Loss
        self.loss.reset()
        for step, step_output_logits in enumerate(seq_output_logits):
            self.loss.eval_step_seq(step_output_logits, seq_output_targets[:, step])
        # print(label_output_logits.size())
        self.loss.eval_label(label_output_logits, label_outputs)
        # Backward propagation
        self.model.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        total_loss, ppl, label_loss = self.loss.get_loss()
        return total_loss, ppl, label_loss

    def run_evaluate(self, eval_set):
        device = None if torch.cuda.is_available() else -1

        eval_iter = torchtext.data.BucketIterator(
            dataset=eval_set, batch_size=self.batch_size,
            sort=True, sort_within_batch=True,
            sort_key=lambda x: len(getattr(x, SRC_WORD_FIELD)),
            device=device, repeat=False, train=True)

        self.model.eval()

        self.loss.reset()

        intent_accuracy, intent_norm = 0, 0
        slot_accuracy, slot_norm = 0, 0
        for batch_data in eval_iter.__iter__():
            # get fields data
            word_inputs, input_lengths = getattr(batch_data, SRC_WORD_FIELD)
            tag1_inputs = getattr(batch_data, SRC_TAG1_FIELD)
            tag2_inputs = getattr(batch_data, SRC_TAG2_FIELD)
            seq_outputs = getattr(batch_data, TGT_SLOT_FIELD)
            label_outputs = getattr(batch_data, TGT_INTENT_FIELD)

            src_feat_inputs = {SRC_WORD_FIELD: word_inputs,
                               SRC_TAG1_FIELD: tag1_inputs,
                               SRC_TAG2_FIELD: tag2_inputs}

            tgt_var_inputs = seq_outputs[:, 0]
            seq_output_targets = seq_outputs[:, 1:]

            # Forward propagation
            label_output_logits, seq_output_logits, res_dict = self.model(src_feat_inputs, input_lengths,
                                                                          tgt_var_inputs)

            slot_seqlist = res_dict['sequence_symbols']
            intent_ids = res_dict['predict_intents']
            # evaluate loss
            for step, step_output_logits in enumerate(seq_output_logits):
                self.loss.eval_step_seq(step_output_logits, seq_output_targets[:, step])

            self.loss.eval_label(label_output_logits, label_outputs)

            exact_match, seq_num = evaluate_slot_filling(eval_set.fields[TGT_SLOT_FIELD].vocab,
                                                         seq_output_targets.data.tolist(),
                                                         slot_seqlist.data.tolist(),
                                                         input_lengths.tolist())
            slot_accuracy += exact_match
            slot_norm += seq_num
            # evaluate accuracy
            intent_accuracy += intent_ids.data.eq(label_outputs.data).sum()

            intent_norm += intent_ids.data.size(0)

        # reset to train mode
        self.model.train()

        total_loss, ppl, label_loss = self.loss.get_loss()
        label_acc = intent_accuracy / intent_norm

        assert intent_norm == slot_norm
        slot_acc = slot_accuracy / slot_norm
        return total_loss, (ppl, slot_acc), (label_loss, label_acc)

    def _train_epochs(self, num_epochs, train_set, valid_set=None):
        device = None if torch.cuda.is_available() else -1

        train_iter = torchtext.data.BucketIterator(
            dataset=train_set, batch_size=self.batch_size,
            sort=True, sort_within_batch=True,
            sort_key=lambda x: len(getattr(x, SRC_WORD_FIELD)),
            device=device, repeat=False)

        # use train mode
        self.model.train()
        total_steps = len(train_iter) * num_epochs

        step = 0
        total_loss, ppl_loss, label_loss = 0.0, 0.0, 0.0
        for epoch in range(num_epochs):
            for batch_data in train_iter.__iter__():
                step_total_loss, step_ppl, step_label_loss = self._train_batch(batch_data)

                step += 1

                total_loss += step_total_loss
                ppl_loss += step_ppl
                label_loss += step_label_loss

                if step % self.display_freq == 0:
                    log_msg = 'Progress: %d%%, Total Loss: %.4f, Slot PPL: %.4f, Intent Loss: %.4f' % \
                              (step / total_steps * 100,
                               total_loss / self.display_freq,
                               ppl_loss / self.display_freq,
                               label_loss / self.display_freq)
                    logging.info(log_msg)
                    total_loss, ppl_loss, label_loss = 0.0, 0.0, 0.0

                # if step % self.ckpt_freq == 0 and valid_set is not None:
            valid_total_loss, (valid_seq_ppl, valid_seq_acc), (valid_label_loss, valid_label_acc) = \
                self.run_evaluate(valid_set)
            log_msg = 'Finished epoch %d:, Dev Total Loss: %.4f; Slot PPL: %.4f, ACC: %.4f, ' \
                      'Intent Loss: %.4f, ACC: %.4f ' % \
                      (epoch + 1,
                       valid_total_loss,
                       valid_seq_ppl, valid_seq_acc,
                       valid_label_loss, valid_label_acc)
            logging.info(log_msg)

        pass

    def train(self, model, optimizer, num_epochs, train_set, valid_set=None):
        self.model = model
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.model.cuda()

        self._train_epochs(num_epochs, train_set, valid_set)
        pass
