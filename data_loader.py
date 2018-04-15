# -*- coding:utf-8 -*-
import logging
import torchtext
import collections


SRC_WORD_FIELD = 'src_word'
SRC_TAG1_FIELD = 'src_tag1'
SRC_TAG2_FIELD = 'src_tag2'
TGT_SLOT_FIELD = 'tgt_slot'
TGT_INTENT_FIELD = 'tgt_intent'


class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        # if kwargs.get('include_lengths') is False:
        #     logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        # kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)


class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.
    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if not kwargs.get('batch_first'):
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]


def get_dataset(train_path, valid_path=None):
    field_src_word = SourceField(include_lengths=True)
    field_src_tag1 = SourceField(include_lengths=False)
    field_src_tag2 = SourceField(include_lengths=False)
    field_tgt_slot = TargetField()
    field_tgt_intent = SourceField(sequential=False)

    train_set = torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=[(SRC_WORD_FIELD, field_src_word),
                (SRC_TAG1_FIELD, field_src_tag1),
                (SRC_TAG2_FIELD, field_src_tag2),
                (TGT_SLOT_FIELD, field_tgt_slot),
                (TGT_INTENT_FIELD, field_tgt_intent)])

    field_src_word.build_vocab(train_set)
    field_src_tag1.build_vocab(train_set)
    field_src_tag2.build_vocab(train_set)
    field_tgt_slot.build_vocab(train_set)
    field_tgt_intent.build_vocab(train_set)

    valid_set = None
    if valid_path is not None:
        valid_set = torchtext.data.TabularDataset(
            path=valid_path, format='tsv',
            fields=[(SRC_WORD_FIELD, field_src_word),
                    (SRC_TAG1_FIELD, field_src_tag1),
                    (SRC_TAG2_FIELD, field_src_tag2),
                    (TGT_SLOT_FIELD, field_tgt_slot),
                    (TGT_INTENT_FIELD, field_tgt_intent)])

    # add fields, must be ordered
    fields_dict = collections.OrderedDict()
    fields_dict[SRC_WORD_FIELD] = field_src_word
    fields_dict[SRC_TAG1_FIELD] = field_src_tag1
    fields_dict[SRC_TAG2_FIELD] = field_src_tag2
    fields_dict[TGT_SLOT_FIELD] = field_tgt_slot
    fields_dict[TGT_INTENT_FIELD] = field_tgt_intent
    return train_set, valid_set, fields_dict
    pass
