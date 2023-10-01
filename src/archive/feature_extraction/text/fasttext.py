#!/usr/bin/python -w

# ------------------------- NOTICE ------------------------------- #
#                                                                  #
#                   CONFIDENTIAL INFORMATION                       #
#                   ------------------------                       #
#     This Document contains Confidential Information or           #
#     Trade Secrets, or both, which are the property of VuNet      #
#     Systems Ltd.  This document may not be copied, reproduced,   #
#     reduced to any electronic medium or machine readable form    #
#     or otherwise duplicated and the information herein may not   #
#     be used, disseminated or otherwise disclosed, except with    #
#     the prior written consent of VuNet Systems Ltd.              #
#                                                                  #
# ------------------------- NOTICE ------------------------------- #

# Copyright 2022 VuNet Systems Ltd.
# All rights reserved.
# Use of copyright notice does not imply publication.

"""
Description : This file adapts fastText word embedding generators and aggregators to
              a common easy to use api in the context of log anomaly detection
Author      : Vunet team
License     : Commercial
"""

import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext

from gensim.models import Word2Vec, KeyedVectors

from preprocessing.tokenize import EventId2Tokens, build_nonoverlapping_sequence
from feature_extraction.fasttext_utils import \
    generate_model_file_path, fast_text_from_model_file, \
    generate_word2vec_model_file_path


def _tokenize_by_spaces(sentence):
    # removes multiple spaces in sentence
    list_with_spaces = str(sentence).strip().replace('\n','').split(" ")
    list_without_spaces = [ele for ele in list_with_spaces if ele.strip()]
    return list_without_spaces


class UnsupervisedEmbedding(ABC):
    def __init__(self, embedding_root: str = None, embedding_for: str = None,
                 embedding_dim: int = 100, embedding_version: float = 1.0):
        self.embedding_root = embedding_root
        self.embedding_for = embedding_for
        self.embedding_dim = embedding_dim
        self.embedding_version = embedding_version

    @abstractmethod
    def fit(self, X):
        pass


class FastTextEmbedding(UnsupervisedEmbedding):
    def __init__(self, embedding_root: str = None, embedding_for: str = None,
                 embedding_model='skipgram', embedding_wordNgrams: int = 1,
                 embedding_dim: int = 100, embedding_version: float = 1.0,
                 epochs: int = 10, minCount: int = 1, maxn: int = 0):
        super().__init__(embedding_root, embedding_for, embedding_dim, embedding_version)
        self.embedding_model = embedding_model

        # Can use wordN grams by setting 2
        # https://fasttext.cc/docs/en/supervised-tutorial.html
        self.embedding_wordNgrams = embedding_wordNgrams

        self.epochs = epochs
        self.minCount = minCount
        self.maxn = maxn

        self.embedding_config = {
            'embedding_root': self.embedding_root,
            'embedding_for': self.embedding_for, 'embedding_model': self.embedding_model,
            "embedding_wordNgrams": self.embedding_wordNgrams, 'embedding_dim': self.embedding_dim,
            'embedding_version': self.embedding_version, 'embedding_type': 'fasttext'
        }

    def fit(self, X):
        seq_len = X.shape[1]
        # if X is not None:
        #     seq_len = len(_tokenize_by_spaces(X[0])) 
        # print(f"Calc BBBBBBBBBB {seq_len} , X.shape[1] = {X.shape[1]}")


        data_temp_file_path = FastTextEmbedding.generate_temp_seq_storage_file_path(self.embedding_for)
        np.savetxt(data_temp_file_path, X.astype(int), fmt='%i')

        # Create embeddings for event id https://fasttext.cc/docs/en/python-module.html
        fasttext_model = fasttext.train_unsupervised(data_temp_file_path,
                                                     model=self.embedding_model,
                                                     dim=self.embedding_dim,
                                                     wordNgrams=self.embedding_wordNgrams,
                                                     epoch=self.epochs, minCount=self.minCount, maxn=self.maxn)
        cfg_copy = self.embedding_config.copy()
        cfg_copy["train_seq_len"] = seq_len
        model_file_path = generate_model_file_path(**cfg_copy)
        fasttext_model.save_model(model_file_path)

        os.remove(data_temp_file_path)

        # print(fasttext_model.get_words())
        # word_embeddings = model.get_output_matrix()
        # print(word_embeddings)

        return fasttext_model

    @staticmethod
    def generate_temp_seq_storage_file_path(embedding_for=None):
        return embedding_for + '_eventid_token_seq.txt'


class FastTextFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, feature_for: str, feature_root: str,
                 vocab_root: str, vocab_version: float = 1.0,
                 feature_model: str = "skipgram", feature_wordNgrams: int = 3,
                 feature_embedding_dim: int = 100, feature_version: float = 1.0,
                 training_sequence_length: int = 10, eventid_colname: str = "event_id",
                 depth: int = 7, st: float = 0.8,
                 eventId2Tokens: EventId2Tokens = None):
        self.feature_for = feature_for
        self.feature_root = feature_root
        self.vocab_root = vocab_root
        self.vocab_version = vocab_version
        self.feature_model = feature_model
        self.feature_wordNgrams = feature_wordNgrams
        self.feature_embedding_dim = feature_embedding_dim
        self.feature_version = feature_version
        self.training_sequence_length = training_sequence_length
        self.eventid_colname = eventid_colname
        self.depth = depth
        self.st = st
        self.eventId2Tokens = eventId2Tokens

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        df = X.copy()

        # NOTE: If this immediate next part of code gives you error,
        # it is because you have not generated the vocab pickle file earlier
        # Run notebook C2_drain_and_postprocessing... to create the vocab
        # In a pipeline, this will generally be taken care of by putting vocab pickle generation ahead of this step

        if self.eventId2Tokens is None:
            pkl_config = {
                'pickle_file_for': self.feature_for, 'pickle_file_version': self.vocab_version,
                'depth': self.depth, 'st': self.st
            }
            self.eventId2Tokens = EventId2Tokens.from_pickle_file(vocab_root=self.vocab_root, **pkl_config)

        eventid_tokens = self.eventId2Tokens.transform(df[self.eventid_colname])
        # print(f"eventid_tokens[0:10]={eventid_tokens[0:10]}")
        # print(f"type(eventid_tokens[0])={type(eventid_tokens[0])}")

        # this will be later needed during transform
        # unique_eventid_tokens = list(set(eventid_tokens)) # this does not preserve the order
        unique_eventid_tokens = list(OrderedDict.fromkeys(eventid_tokens))  # this preserves order of original list
        # print(f"unique eventid tokens = {unique_eventid_tokens}")

        eventid_token_seqs = build_nonoverlapping_sequence(eventid_tokens, seq_len=self.training_sequence_length)
        # print(eventid_token_seqs.shape)
        # print(eventid_token_seqs[0:5])

        fastText_embedding_cfg = {
            'embedding_root': self.feature_root,
            'embedding_for': self.feature_for, 'embedding_model': self.feature_model,
            "embedding_wordNgrams": self.feature_wordNgrams, 'embedding_dim': self.feature_embedding_dim,
            'embedding_version': self.feature_version
        }

        fasttext_embedding = FastTextEmbedding(**fastText_embedding_cfg)
        fasttext_model = fasttext_embedding.fit(eventid_token_seqs) # fits and saves model

        fasttext_model = fast_text_from_model_file(**fastText_embedding_cfg)
        self.word_embeddings = np.array([fasttext_model.get_word_vector(str(word_token))
                                         for word_token in unique_eventid_tokens])

        # self.normalized_word_embeddings = \
        #    np.array([embedding/np.linalg.norm(embedding) for embedding in self.word_embeddings])
        return self

    # transform does not depend on input X
    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        return self.word_embeddings


class FastTextFeatureLoader(BaseEstimator, TransformerMixin):
    def __init__(self, feature_for: str, feature_root: str,
                 vocab_root: str, vocab_version: float = 1.0,
                 feature_model: str = "skipgram", feature_wordNgrams: int = 3,
                 feature_embedding_dim: int = 100, feature_version: float = 1.0,
                 training_sequence_length: int = 10, eventid_colname: str = "event_id",
                 depth: int = 7, st: float = 0.8,
                 load_normalized_embeddings: bool = False,
                 eventId2Tokens: EventId2Tokens = None):
        self.feature_for = feature_for
        self.feature_root = feature_root
        self.vocab_root = vocab_root
        self.vocab_version = vocab_version
        self.feature_model = feature_model
        self.feature_wordNgrams = feature_wordNgrams
        self.feature_embedding_dim = feature_embedding_dim
        self.feature_version = feature_version
        self.training_sequence_length = training_sequence_length
        self.eventid_colname = eventid_colname
        self.depth = depth
        self.st = st
        self.load_normalized_embeddings = load_normalized_embeddings
        self.eventId2Tokens = eventId2Tokens

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    # X is not required since this is going to be ingest process done outside pipeline
    def transform(self, X: pd.DataFrame = None, y: pd.Series = None):
        df = X
        if self.eventId2Tokens is None:
            pkl_config = {
                'pickle_file_for': self.feature_for, 'pickle_file_version': self.vocab_version,
                'depth': self.depth, 'st': self.st
            }
            self.eventId2Tokens = EventId2Tokens.from_pickle_file(vocab_root=self.vocab_root, **pkl_config)

        eventid_tokens = self.eventId2Tokens.transform(df[self.eventid_colname])

        # None of these are needed when df is already unique
        # unique_eventid_tokens = list(set(eventid_tokens)) # this does not preserve the order
        # unique_eventid_tokens = list(OrderedDict.fromkeys(eventid_tokens)) # this preserves order of original list
        # print(f"unique eventid tokens = {unique_eventid_tokens}")
        self.unique_event_id_tokens = eventid_tokens
        self.unique_event_ids = df[self.eventid_colname]
        
        fasttext_embedding_cfg = {
            'embedding_root': self.feature_root,
            'embedding_for': self.feature_for, 'embedding_model': self.feature_model,
            "embedding_wordNgrams": self.feature_wordNgrams, 'embedding_dim': self.feature_embedding_dim,
            'embedding_version': self.feature_version
            # 'minCount':1, 'maxn': 0
        }

        fasttext_model = fast_text_from_model_file(**fasttext_embedding_cfg)
        word_embeddings = np.array([fasttext_model.get_word_vector(str(word_token))
                                    for word_token in self.unique_event_id_tokens])

        if self.load_normalized_embeddings is True:
            normalized_word_embeddings = \
                np.array([embedding/np.linalg.norm(embedding) for embedding in word_embeddings])
            return normalized_word_embeddings
        else:
            return word_embeddings

class FastTextWord2VecFeatureLoader(BaseEstimator, TransformerMixin):
    def __init__(self, feature_for: str, feature_root: str,
                 vocab_root: str, vocab_version: float = 1.0,
                 feature_model: str = "skipgram", feature_wordNgrams: int = 3,
                 feature_embedding_dim: int = 100, feature_version: float = 1.0,
                 training_sequence_length: int = 10, eventid_colname: str = "event_id",
                 depth: int = 7, st: float = 0.8,
                 load_normalized_embeddings: bool = False,
                 eventId2Tokens: EventId2Tokens = None):
        self.feature_for = feature_for
        self.feature_root = feature_root
        self.vocab_root = vocab_root
        self.vocab_version = vocab_version
        self.feature_model = feature_model
        self.feature_wordNgrams = feature_wordNgrams
        self.feature_embedding_dim = feature_embedding_dim
        self.feature_version = feature_version
        self.training_sequence_length = training_sequence_length
        self.eventid_colname = eventid_colname
        self.depth = depth
        self.st = st
        self.load_normalized_embeddings = load_normalized_embeddings
        self.eventId2Tokens = eventId2Tokens

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    # X is not required since this is going to be ingest process done outside pipeline
    def transform(self, X: pd.DataFrame = None, y: pd.Series = None):
        df = X
        if self.eventId2Tokens is None:
            pkl_config = {
                'pickle_file_for': self.feature_for, 'pickle_file_version': self.vocab_version,
                'depth': self.depth, 'st': self.st
            }
            self.eventId2Tokens = EventId2Tokens.from_pickle_file(vocab_root=self.vocab_root, **pkl_config)

        eventid_tokens = self.eventId2Tokens.transform(df[self.eventid_colname])

        # None of these are needed when df is already unique
        # unique_eventid_tokens = list(set(eventid_tokens)) # this does not preserve the order
        # unique_eventid_tokens = list(OrderedDict.fromkeys(eventid_tokens)) # this preserves order of original list
        # print(f"unique eventid tokens = {unique_eventid_tokens}")
        self.unique_event_id_tokens = eventid_tokens
        self.unique_event_ids = df[self.eventid_colname]
        
        fasttext_embedding_cfg = {
            'embedding_root': self.feature_root,
            'embedding_for': self.feature_for, 'embedding_model': self.feature_model,
            "embedding_wordNgrams": self.feature_wordNgrams, 'embedding_dim': self.feature_embedding_dim,
            'embedding_version': self.feature_version,
            "embedding_type": "fasttext"
            # 'minCount':1, 'maxn': 0
        }

        model_file_path = generate_word2vec_model_file_path(**fasttext_embedding_cfg)

        wv = KeyedVectors.load_word2vec_format(model_file_path)
        wv_eventids = wv.index_to_key
        # print(wv_eventids)
        word_embeddings = np.zeros((len(wv_eventids), self.feature_embedding_dim))
        # for i, eventid in enumerate(self.eventids):
        #    X_out[i] = wv[eventid].reshape(1, -1)
        for i, eventid in enumerate(self.unique_event_ids):
            # print(f"eventid={eventid}")
            word_embeddings[i] = wv[eventid].reshape(1, -1)

        if self.load_normalized_embeddings is True:
            normalized_word_embeddings = \
                np.array([embedding/np.linalg.norm(embedding) for embedding in word_embeddings])
            return normalized_word_embeddings
        else:
            return word_embeddings
