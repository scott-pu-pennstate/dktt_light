import os
import math
from collections import defaultdict
import logging

from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from dktt_light_config import NAME_CONVENTION, VERBOSE

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO if VERBOSE == 1 else logging.WARNING)
console = logging.StreamHandler()
console.setLevel(logging.INFO if VERBOSE == 1 else logging.WARNING)
logger.addHandler(console)


class Encoder:
    r"""a simple encoder"""
    def __init__(self, vocab):
        # add special tokens to list
        self.vocab = ['<pad>', '<unk>'] + sorted(vocab)

        # create vocab related dicts
        self.dictionary = defaultdict(lambda: 1)
        self.dictionary.update(dict(
            zip(self.vocab, range(len(self.vocab)))))

    def encode(self, word):
        return self.dictionary[word]


class DataSet:
    def __init__(
            self,
            dataset,
            data_dir,
            cv_idx,
            id_col,
            time_col,
            prob_col,
            skill_col,
            score_col,
            skill_sep,
            max_len,
            **kwargs
    ):
        r"""configure the dataset"""
        self.dataset = dataset
        self.id_col = id_col
        self.time_col = time_col
        self.prob_col = prob_col
        self.skill_col = skill_col
        self.score_col = score_col
        self.skill_sep = skill_sep
        self.max_len = max_len
        self.dataset = dataset

        self.train_fpath = os.path.join(
            data_dir,
            NAME_CONVENTION.format(dataset, 'train', cv_idx))
        self.test_fpath = os.path.join(
            data_dir,
            NAME_CONVENTION.format(dataset, 'test', cv_idx))

    def describe_data(self):
        df = pd.concat([self.train_df, self.test_df], axis=0)
        num_attempts = df.shape[0]
        num_students = df[self.id_col].unique().shape[0]
        if self.dataset == 'stat2011':
            num_skills = df[self.prob_col].unique().shape[0]
        else:
            num_skills = df[self.skill_col].unique().shape[0]
        num_items = df[self.prob_col].unique().shape[0]

        logger.info(f'dataset = {self.dataset}')
        logger.info(f'num of attempts = {num_attempts / 1000} K')
        logger.info(f'num of students = {num_students}')
        logger.info(f'num of skills = {num_skills}')
        logger.info(f'num of items = {num_items}')

    def load_data(self):
        self.train_df = pd.read_csv(self.train_fpath)
        logger.info(f'train data in {self.train_fpath} is loaded')
        self.test_df = pd.read_csv(self.test_fpath)
        logger.info(f'test data in {self.test_fpath} is loaded')

    def preprocess(self):
        if self.dataset == 'stat2011':
            self.train_df[self.time_col] = pd.to_datetime(self.train_df[self.time_col])
            self.train_df[self.skill_col] = self.train_df[self.prob_col]

            self.test_df[self.time_col] = pd.to_datetime(self.test_df[self.time_col])
            self.test_df[self.skill_col] = self.test_df[self.prob_col]

        prob_encoder, skill_encoder = self.get_encoder()
        logger.info(f'problem and skill encoders are created')

        for df in [self.train_df, self.test_df]:
            # combine prob and score; skill and score
            df['prob_and_score'] = df.apply(
                lambda x: (x[self.prob_col], x[self.score_col]), axis=1)
            df['prob_and_pscore'] = df.apply(
                lambda x: (x[self.prob_col], 1), axis=1)
            df['skill_and_score'] = df.apply(
                lambda x: (x[self.skill_col], x[self.score_col]), axis=1)

            # encode
            df['prob_and_score'] = df['prob_and_score'].apply(prob_encoder.encode)
            df['prob_and_pscore'] = df['prob_and_pscore'].apply(prob_encoder.encode)
            df['skill_and_score'] = df['skill_and_score'].apply(skill_encoder.encode)
            df[self.score_col] = df[self.score_col] + 1
        logger.info('train and test data are cleaned')

        train_features = self.train_df.groupby(self.id_col).apply(self.extract_features)
        test_features = self.test_df.groupby(self.id_col).apply(self.extract_features)
        logger.info('train and test sequences are extracted')

        cols = ['prob_and_score', 'prob_and_pscore', self.score_col, self.time_col]
        dtypes = ['int32', 'int32', 'int32', 'float32']
        train_hist_folds, train_next_folds = self.get_folded_seqs(train_features[cols].values, dtypes)
        test_hist_folds, test_next_folds = self.get_folded_seqs(test_features[cols].values, dtypes)
        logger.info('train and test sequences are folded')

        train_inputs, train_targets = self.get_inputs_and_targets_from_folds(
            train_hist_folds, train_next_folds)
        test_inputs, test_targets = self.get_inputs_and_targets_from_folds(
            test_hist_folds, test_next_folds)
        logger.info('train and test inputs and targets are created')

        item_skill_mapping = self.get_item_skill_mapping(prob_encoder, skill_encoder)
        logger.info('problem skill mapping is created')

        return (train_inputs, train_targets), (test_inputs, test_targets), item_skill_mapping

    def extract_features(self, df, time_unit=3600.):
        df[self.time_col] = (df[self.time_col] - df[self.time_col].min()).apply(
            lambda x: x if type(x) in set([int, float]) else x.total_seconds())  # time difference must be either numeric or time_delta
        df = df.sort_values(self.time_col)

        # flatten df, initialize with id and time_col
        df[self.time_col] = df[self.time_col] / time_unit

        flattened = []
        for col in ['prob_and_score', 'prob_and_pscore', self.score_col, self.time_col]:
            seq = df[col].values.tolist()
            flattened.append((col, [seq]))

        return pd.DataFrame(dict(flattened))

    def get_encoder(self):
        # vocab is based on all data, assuming that item-skill is known to the system before any user data
        prob_vocab = self.get_vocab(pd.concat([self.train_df, self.test_df], axis=0), self.prob_col)
        skill_vocab = self.get_vocab(pd.concat([self.train_df, self.test_df], axis=0), self.skill_col)
        prob_encoder = Encoder(prob_vocab)
        skill_encoder = Encoder(skill_vocab)
        return prob_encoder, skill_encoder

    def get_vocab(self, data, col):
        vocab = data[col].unique()
        vocab_w = [(v, 0) for v in vocab]
        vocab_r = [(v, 1) for v in vocab]
        return vocab_w + vocab_r

    def fold_seq(self, seq):
        hist_seqs, next_seqs = [], []
        num_folds = math.ceil(len(seq) / self.max_len)

        for idx in range(num_folds):
            next_start = idx * self.max_len
            next_end = next_start + self.max_len

            hist_start = next_start
            hist_end = min(next_end - 1, len(seq) - 1)

            next_seq = seq[next_start: next_end]
            hist_seq = seq[hist_start: hist_end]

            next_seqs.append(next_seq)
            hist_seqs.append(hist_seq)

        return hist_seqs, next_seqs

    def get_folded_seqs(self, df, dtypes):
        hist_folds = [[] for _ in range(df.shape[1])]
        next_folds = [[] for _ in range(df.shape[1])]

        for row in tqdm(range(df.shape[0])):
            for idx, seq in enumerate(df[row]):
                col_hist_seq, col_next_seq = self.fold_seq(seq)
                hist_folds[idx].extend(col_hist_seq)
                next_folds[idx].extend(col_next_seq)

        hist_padded_folds = [tf.keras.preprocessing.sequence.pad_sequences(folds, maxlen=self.max_len, dtype=dtype) for folds, dtype in zip(hist_folds, dtypes)]
        next_padded_folds = [tf.keras.preprocessing.sequence.pad_sequences(folds, maxlen=self.max_len, dtype=dtype) for folds, dtype in zip(next_folds, dtypes)]
        return hist_padded_folds, next_padded_folds

    @staticmethod
    def get_inputs_and_targets_from_folds(hist_folds, next_folds):
        inputs = {
            'encoder_items': hist_folds[0],
            'encoder_times': hist_folds[3],
            'decoder_items': next_folds[1],
            'decoder_times': next_folds[3]}

        targets = next_folds[2]

        return inputs, targets

    def get_item_skill_mapping(self, prob_encoder, skill_encoder):
        prob_and_skills = self.train_df[[self.prob_col, self.skill_col]]
        agg_prob_and_skills = prob_and_skills.groupby(self.prob_col)[self.skill_col].apply(
            lambda x: sorted(list(set(x))))

        q_matrix = np.zeros((len(prob_encoder.vocab), len(skill_encoder.vocab)))

        for prob, skills in agg_prob_and_skills.items():
            # encode prob
            encoded_prob_w = prob_encoder.encode((prob, 0))
            encoded_prob_r = prob_encoder.encode((prob, 1))

            for skill in skills:
                # encode skill
                encoded_skill_w = skill_encoder.encode((skill, 0))
                encoded_skill_r = skill_encoder.encode((skill, 1))

                # add prob-skill association to the q-matrix
                q_matrix[encoded_prob_w, encoded_skill_w] = 1.
                q_matrix[encoded_prob_r, encoded_skill_r] = 1.

        return q_matrix
