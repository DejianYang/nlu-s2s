# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        # self.linear_out = nn.Linear(dim * 2, dim)

    @staticmethod
    def _sequence_mask(lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return 1 - (torch.arange(0, max_len)
                    .type_as(lengths)
                    .repeat(batch_size, 1)
                    .lt(lengths.unsqueeze(1)))

    def forward(self, hidden, memory, memory_length=None):
        seq_length = memory.size(1)
        # (batch*1*dim)*(batch*dim*seq_l) -> batch*1*seq_l = batch*seq_l
        attn = torch.bmm(hidden.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)

        if memory_length is not None:
            mask = self._sequence_mask(memory_length, seq_length)
            attn.data.masked_fill_(mask, -float('inf'))

        # batch * seq_l
        attn = F.softmax(attn, dim=1)
        # (batch, 1, seq_l) * (batch, seq_l, dim) -> (batch, 1, dim) -> (batch, dim)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
        return context, attn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_idx=0, input_dropout_p=0.0):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.pad_idx = pad_idx
        self.init_dropout_p = input_dropout_p
        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        pass

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.embeddings.weight.data.copy_(pretrained)
            if fixed:
                self.embeddings.weight.requires_grad = False

    def forward(self, input_vars):
        emb_inputs = self.embeddings(input_vars)
        emb_inputs = self.input_dropout(emb_inputs)
        return emb_inputs


class BaseRNN(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, n_layers=1, bidirectional=False, dropout_p=0.0):
        super(BaseRNN, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size

        if bidirectional:
            assert hidden_size % 2 == 0
        rnn_hidden_size = hidden_size // 2 if bidirectional else hidden_size

        self.rnn = self.get_rnn_cell(cell_type, input_size, rnn_hidden_size, n_layers, bidirectional, dropout_p)
        pass

    @staticmethod
    def get_rnn_cell(rnn_cell, input_size, hidden_size, n_layers, bidirectional=False, dropout_p=0.0):
        if rnn_cell.lower() == 'lstm':
            return nn.LSTM(input_size, hidden_size, n_layers,
                           batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        elif rnn_cell.lower() == 'gru':
            return nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            raise NotImplementedError("No such RNN cell: {}".format(rnn_cell))

    def rnn_forward(self, inputs, input_lengths=None):
        """
        dynamic rnn run
        :param inputs: embedding inputs, shape=[batch_size, sequence_length], type=torch.LongTensor
        :param input_lengths: sorted inputs lengths, shape=[batch_size], type=torch.LongTensor
        :return:
        """
        emb_inputs = inputs
        if input_lengths is not None:
            emb_inputs = nn.utils.rnn.pack_padded_sequence(emb_inputs, input_lengths.tolist(), batch_first=True)
        outputs, hidden = self.rnn(emb_inputs)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class EncoderRNN(BaseRNN):
    def __init__(self, features, hidden_size, rnn_cell='gru', n_layers=1, bidirectional=False, dropout_p=0.0):
        """
        RNN Encoder with embeddings
        :param features: features dict, including words, or pos_tag
        :param hidden_size: hidden size of rnn cell
        :param rnn_cell: rnn cell type, LSTM or GRU
        :param n_layers: number of layers
        :param bidirectional: bidirectional
        :param dropout_p: rnn cell drop output
        """

        assert isinstance(features, dict)
        assert len(features) > 0
        self.features_dict = features
        self.feature_embeddings = dict()
        input_size = 0

        input_feature_modules = []
        for feat_name in features:
            feature = features[feat_name]
            feat_emb = Embeddings(vocab_size=feature['vocab_size'],
                                  emb_size=feature['emb_size'],
                                  pad_idx=0,
                                  input_dropout_p=feature['dropout_p'])
            input_size += feature['emb_size']

            if torch.cuda.is_available():
                feat_emb.cuda()
            input_feature_modules += [(feat_name, feat_emb)]

        super(EncoderRNN, self).__init__(rnn_cell, input_size, hidden_size,
                                         n_layers=n_layers, bidirectional=bidirectional, dropout_p=dropout_p)

        self.feature_embeddings = nn.Sequential(OrderedDict(input_feature_modules))
        pass

    def forward(self, feat_inputs, input_lengths=None):
        """
        :param feat_inputs: input ids for each feature, dict
        :param input_lengths: input length
        :return:
        """
        rnn_inputs = []
        assert len(feat_inputs) == len(self.feature_embeddings)
        for feat_name in feat_inputs:
            feat_input_vars = feat_inputs[feat_name]
            rnn_inputs += [getattr(self.feature_embeddings, feat_name).forward(feat_input_vars)]
        # concat all feature embeddings
        rnn_inputs = torch.cat(rnn_inputs, dim=2)
        return self.rnn_forward(rnn_inputs, input_lengths)
        pass


class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_SEQUENCE = 'sequence_symbols'

    def __init__(self, vocab_size, emb_size, hidden_size, max_length, input_dropout_p=0.0,
                 rnn_cell='lstm', n_layers=1, bidirectional=False, dropout_p=0.0,
                 use_attention=False):

        self.encoder_bidirectional = bidirectional
        self.max_length = max_length
        self.output_size = vocab_size
        super(DecoderRNN, self).__init__(rnn_cell, hidden_size + emb_size, hidden_size,
                                         n_layers=n_layers, bidirectional=False, dropout_p=dropout_p)

        self.embeddings = Embeddings(vocab_size=vocab_size, emb_size=emb_size, input_dropout_p=input_dropout_p)
        self.use_attention = use_attention

        if self.use_attention:
            self.attention = Attention(dim=hidden_size)
            self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        pass

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.encoder_bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def forward_step(self, input_vars, hidden, memory, memory_lengths=None):
        assert input_vars.dim() == 2
        attn = None
        if isinstance(hidden, tuple):
            rnn_inp = torch.cat((hidden[0][0], input_vars), dim=1).unsqueeze(1)
            _, (h, c) = self.rnn.forward(rnn_inp, hidden)
            h = h.squeeze(0)
            if self.use_attention:
                ctx, attn = self.attention.forward(h, memory, memory_length=memory_lengths)
                h = self.linear_out(torch.cat([h, ctx], dim=1))

            logits = F.log_softmax(self.out(h), dim=1)
            return logits, (h.unsqueeze(0), c), attn

        else:
            rnn_inp = torch.cat((hidden[0][0], input_vars), dim=1).squeeze(1)
            _, h = self.rnn(rnn_inp, hidden)
            h = h.squeeze(0)
            if self.use_attention:
                ctx, attn = self.attention.forward(h, memory, memory_length=memory_lengths)
                h = self.linear_out(torch.cat([h, ctx], dim=1))

            logits = F.log_softmax(self.out(h), dim=1)
            return logits, h.unsqueeze(0), attn
        pass

    def forward(self, input_vars, encoder_hidden=None, encoder_outputs=None, encoder_lengths=None):
        decoder_hidden = self._init_state(encoder_hidden)
        batch_size, step_size = input_vars.size()
        emb_inputs = self.embeddings(input_vars)

        output_logits = []
        attention_dists = []
        sequence_symbols = []

        for step in range(step_size):
            step_input_vars = emb_inputs[:, step, :]
            step_logits, decoder_hidden, attn_dist = self.forward_step(step_input_vars,
                                                                       hidden=decoder_hidden,
                                                                       memory=encoder_outputs,
                                                                       memory_lengths=encoder_lengths)
            symbols = step_logits.topk(1)[1]

            output_logits += [step_logits]
            attention_dists += [attn_dist]
            sequence_symbols += [symbols]

        res_dict = dict()
        res_dict[DecoderRNN.KEY_ATTN_SCORE] = torch.stack(attention_dists, dim=0)
        res_dict[DecoderRNN.KEY_SEQUENCE] = torch.stack(sequence_symbols, dim=0).squeeze(2)

        return output_logits, decoder_hidden, res_dict
        pass


class SeqLabelDecoderRNN(DecoderRNN):
    def __init__(self, vocab_size, emb_size, hidden_size, max_length, input_dropout_p=0.0, rnn_cell='lstm',
                 n_layers=1, bidirectional=False, dropout_p=0.0,
                 use_attention=False):
        super(SeqLabelDecoderRNN, self).__init__(vocab_size, emb_size, hidden_size, max_length, input_dropout_p,
                                                 rnn_cell, n_layers, bidirectional, dropout_p, use_attention)
        self.cat_inputs = nn.Linear(emb_size + hidden_size, emb_size)

    def forward(self, input_vars, encoder_hidden=None, encoder_outputs=None, encoder_lengths=None):
        # inputs only have sos flags
        assert input_vars.dim() == 1
        # init label inputs with sos
        label_vars = input_vars

        # init decoder hidden
        decoder_hidden = self._init_state(encoder_hidden)

        # make sure the decoder steps is equal to the encoder
        max_step = encoder_outputs.size(1)

        output_logits = []
        attention_dists = []
        sequence_symbols = []
        for step in range(max_step):
            encoder_inputs = encoder_outputs[:, step, :]
            label_inputs = self.embeddings(label_vars)
            decoder_inputs = self.cat_inputs(torch.cat((label_inputs, encoder_inputs), dim=1))
            step_logits, decoder_hidden, attn_dist = self.forward_step(decoder_inputs,
                                                                       hidden=decoder_hidden,
                                                                       memory=encoder_outputs,
                                                                       memory_lengths=encoder_lengths)

            symbols = step_logits.topk(1)[1]
            label_vars = symbols.squeeze(1)

            output_logits += [step_logits]
            attention_dists += [attn_dist]
            sequence_symbols += [symbols]

        res_dict = dict()
        res_dict[DecoderRNN.KEY_ATTN_SCORE] = torch.stack(attention_dists, dim=0)
        res_dict[DecoderRNN.KEY_SEQUENCE] = torch.stack(sequence_symbols, dim=1).squeeze(2)

        return output_logits, decoder_hidden, res_dict
        pass


class SlotLabelSeq2Seq(nn.Module):
    KEY_PREDICT_INTENTS = 'predict_intents'

    def __init__(self, encoder, decoder, classifier):
        super(SlotLabelSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        pass

    def forward(self, src_feat_inputs, src_input_lengths, dec_input_vars):
        encoder_outputs, encoder_hidden = self.encoder(src_feat_inputs, src_input_lengths)

        output_logits, decoder_hidden, res_dict = self.decoder(dec_input_vars,
                                                               encoder_hidden,
                                                               encoder_outputs,
                                                               src_input_lengths)

        intent_hidden = encoder_hidden[0] if isinstance(encoder_hidden, tuple) else encoder_hidden
        intent_hidden = torch.cat([intent_hidden[i] for i in range(intent_hidden.size(0))], dim=1)

        # intent detection
        intent_logits = self.classifier(intent_hidden)
        predict_intents = intent_logits.topk(1)[1].squeeze(1)
        res_dict[SlotLabelSeq2Seq.KEY_PREDICT_INTENTS] = predict_intents
        return intent_logits, output_logits, res_dict
        pass
