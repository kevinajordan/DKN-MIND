#!/usr/bin/env python3
import csv
import sys
import os
from tempfile import TemporaryDirectory
import logging
import json
from pathlib import Path
from time import perf_counter

import tensorflow as tf
import numpy as np
import pandas as pd
tf.get_logger().setLevel('ERROR') # only show error messages

from deeprec_utils import download_deeprec_resources, prepare_hparams
from dkn import DKN
from dkn_iterator import DKNTextIterator

from flask import Flask, jsonify, request
from flask_expects_json import expects_json


def get_model(saved_model: str = 'final'):
    print(f"System version: {sys.version}")
    print(f"Tensorflow version: {tf.__version__}")

    # DKN parameters
    history_size = 50

    data_path = Path.cwd() / r'mind-demo-dkn'

    yaml_file = os.path.join(data_path, r'dkn.yaml')
    train_file = os.path.join(data_path, r'train_mind_demo.txt')
    valid_file = os.path.join(data_path, r'valid_mind_demo.txt')
    test_file = os.path.join(data_path, r'test_mind_demo.txt')
    news_feature_file = os.path.join(data_path, r'doc_feature.txt')
    user_history_file = os.path.join(data_path, r'user_history.txt')
    wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
    entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')
    contextEmb_file = os.path.join(data_path, r'TransE_context2vec_100.npy')

    # Create hyperparameters
    print("Creating hyperparameters...")
    hparams = prepare_hparams(
        yaml_file,
        news_feature_file = news_feature_file,
        user_history_file = user_history_file,
        wordEmb_file=wordEmb_file,
        entityEmb_file=entityEmb_file,
        contextEmb_file=contextEmb_file,
        history_size=history_size,
        batch_size=9432//8,  # 9432 unique news items
        save_model=True,
        model_dir=Path.cwd()/'model'
    )

    model = DKN(hparams, DKNTextIterator)

    print("loading saved model:", saved_model)
    mp = (Path(hparams.model_dir) / saved_model)
    model.load_model(str(mp))
    print('model ready')

    return model

def load_news(path):
    news = pd.read_table(
        path, sep="\t", lineterminator='\n', header=None, quoting=csv.QUOTE_NONE,
        index_col='news_id',
        names=['news_id', 'category', 'subcategory', 'title',
               'abstract', 'url', 'title_entities', 'abstract_entities']
    )
    news['title_entities'] = news['title_entities'].map(json.loads)
    news['abstract_entities'] = news['abstract_entities'].map(json.loads)
    return news

def make_app(saved_model: str = 'final'):
    print("Creating model...")
    model = get_model(saved_model)
    print("Loading news...")

    news = pd.concat([load_news(p) for p in (
        "data/test/news.tsv",
        "data/dev/news.tsv",
        "data/train/news.tsv",
    )], join='inner')

    news = news[~news.index.duplicated(keep='first')]
    print("Ready.")
    app = Flask(__name__)

    @app.route("/rank-ids")
    @expects_json()
    def rank_ids():
        """
        Takes a JSON object with arguments
         - articles: [up to hist_size IDs]
         - limit: integer for how many results to return (default 100)

         All news IDs are strings in the form m/N[0-9]+/ .
         The IDs in the array MUST exist in doc_feature.txt
         """
        inputs = request.get_json()
        doc_size = model.hparams.doc_size
        batch_size = model.hparams.batch_size
        hist_size = model.hparams.history_size
        it = model.iterator

        request_keys = inputs['articles'][-hist_size:]
        limit = inputs.get('limit', 100)

        keys = sorted(it.news_word_index.keys())
        subbatch_size = len(keys)//8
        keys_arr = np.array(keys)

        clicki = np.array([[it.news_word_index[k] for k in request_keys]])
        clickei = np.array([[it.news_entity_index[k] for k in request_keys]])
        extra_hist_pad = hist_size - clicki.shape[1]
        clicki = np.pad(clicki, ((0,0), (0, extra_hist_pad), (0,0)), 'constant', constant_values=(0))
        clickei = np.pad(clickei, ((0,0), (0, extra_hist_pad), (0,0)), 'constant', constant_values=(0))

        scores = np.zeros((len(keys)), dtype=np.float32)
        total_t = 0.0

        for key_batch_i in range(8):
            bk = keys[key_batch_i*subbatch_size: (key_batch_i+1)*subbatch_size]

            news_word_rows = np.zeros((subbatch_size, doc_size))
            news_ent_rows = np.zeros((subbatch_size, doc_size))

            for i, k in enumerate(bk):
                news_word_rows[i] = it.news_word_index[k]
                news_ent_rows[i] = it.news_entity_index[k]

            infer_start = perf_counter()

            batch = model.iterator.gen_feed_dict({
                'candidate_news_index_batch': news_word_rows,
                'candidate_news_entity_index_batch': news_ent_rows,
                'click_news_index_batch': np.repeat(clicki, batch_size, 0),
                'click_news_entity_index_batch': np.repeat(clickei, batch_size, 0),
                'labels': np.zeros(()),  # not used by inference

            })

            res = model.infer(model.sess, batch)[0].reshape((-1))
            scores[key_batch_i*subbatch_size: (key_batch_i+1)*subbatch_size] = res

            total_t += perf_counter() - infer_start

        print(f"Inference took {total_t}s")
        sort_idx = scores.argsort()[:-limit + 1:-1]
        ret = keys_arr[sort_idx]

        return jsonify([
            {
                "key": keys_arr[x],
                "score": float(scores[x]),
                "title": news.loc[keys_arr[x]].title,
                "category": news.loc[keys_arr[x]].category,
                "abstract": news.loc[keys_arr[x]].abstract,
            }
            for x in sort_idx
        ])

    @app.route("/rank")
    @expects_json()
    def rank():
        """
        Takes a JSON object with arrays of the following spec:
         - clicks: a nested array containing at least one row of doc_size
                   integers, which are the words from clicked articles.
         - click_ent: a nested array containing at least one row of doc_size
                      integers, which are the entities from clicked articles.

         All indices refer to those used by the news_feature_file and the
         loaded word/entity embeddings. Short histories will be padded, and
         long histories will be truncated to the last history_size entries.
         """

        inputs = request.get_json()
        doc_size = model.hparams.doc_size
        batch_size = model.hparams.batch_size
        hist_size = model.hparams.history_size
        it = model.iterator

        keys = sorted(it.news_word_index.keys())
        subbatch_size = len(keys)//8
        keys_arr = np.array(keys)

        clicki = np.array([inputs['clicks'][-hist_size:]])
        clickei = np.array([inputs['click_ent'][-hist_size:]])
        extra_hist_pad = hist_size - clicki.shape[1]
        clicki = np.pad(clicki, ((0,0), (0, extra_hist_pad), (0,0)), 'constant', constant_values=(0))
        clickei = np.pad(clickei, ((0,0), (0, extra_hist_pad), (0,0)), 'constant', constant_values=(0))

        scores = np.zeros((len(keys)), dtype=np.float32)
        total_t = 0.0

        for key_batch_i in range(8):
            bk = keys[key_batch_i*subbatch_size: (key_batch_i+1)*subbatch_size]

            news_word_rows = np.zeros((subbatch_size, doc_size))
            news_ent_rows = np.zeros((subbatch_size, doc_size))

            for i, k in enumerate(bk):
                news_word_rows[i] = it.news_word_index[k]
                news_ent_rows[i] = it.news_entity_index[k]

            infer_start = perf_counter()

            batch = model.iterator.gen_feed_dict({
                'candidate_news_index_batch': news_word_rows,
                'candidate_news_entity_index_batch': news_ent_rows,
                'click_news_index_batch': np.repeat(clicki, batch_size, 0),
                'click_news_entity_index_batch': np.repeat(clickei, batch_size, 0),
                'labels': np.zeros(()),  # not used by inference

            })

            res = model.infer(model.sess, batch)[0].reshape((-1))
            scores[key_batch_i*subbatch_size: (key_batch_i+1)*subbatch_size] = res

            total_t += perf_counter() - infer_start

        print(f"Inference took {total_t}s")
        sort_idx = scores.argsort()[:-101:-1]
        ret = keys_arr[sort_idx]

        return jsonify([
            {
                "key": keys_arr[x],
                "score": float(scores[x]),
                "title": news.loc[keys_arr[x]].title,
                "category": news.loc[keys_arr[x]].category,
                "abstract": news.loc[keys_arr[x]].abstract,
            }
            for x in sort_idx
        ])


    @app.route("/infer")
    @expects_json()
    def infer():
        """
        Takes a JSON object with arrays of the following spec:
         - words: an array of doc_size integers, which are word indices
           of the article being checked for relevance.
         - entities: an array of doc_size integers, which are entity indices
                     of the article being checked for relevance.
         - clicks: a nested array containing at least one row of doc_size
                   integers, which are the words from clicked articles.
         - click_ent: a nested array containing at least one row of doc_size
                      integers, which are the entities from clicked articles.

         All indices refer to those used by the news_feature_file and the
         loaded word/entity embeddings. Short histories will be padded, and
         long histories will be truncated to the last history_size entries.
        """
        doc_size = model.hparams.doc_size
        batch_size = model.hparams.batch_size
        hist_size = model.hparams.history_size
        it = model.iterator

        inputs = request.get_json()
        cni = np.array([inputs['words']])
        cnei = np.array([inputs['entities']])

        clicki = np.array([inputs['clicks'][-hist_size:]])
        clickei = np.array([inputs['click_ent'][-hist_size:]])
        extra_hist_pad = hist_size - clicki.shape[1]

        batch = model.iterator.gen_feed_dict({
            'candidate_news_index_batch': np.pad(cni, ((0, batch_size-1), (0,0)), 'constant', constant_values=(0)),
            'candidate_news_entity_index_batch': np.pad(cnei, ((0, batch_size-1), (0,0)), 'constant', constant_values=(0)),
            'click_news_index_batch': np.pad(clicki, ((0, batch_size-1), (0, extra_hist_pad), (0,0)), 'constant', constant_values=(0)),
            'click_news_entity_index_batch': np.pad(clickei, ((0, batch_size-1), (0, extra_hist_pad), (0,0)), 'constant', constant_values=(0)),
            'labels': np.zeros(()),  # not used by inference
        })

        del batch[model.iterator.labels]

        res = model.infer(model.sess, batch)[0][0][0]
        return {"score": float(res)}


    @app.route("/get-embedding")
    @expects_json()
    def get_embedding():
        """
        Takes a JSON object with arrays of the following spec:
         - words: an array of doc_size integers, which are word indices
         - entities: an array of doc_size integers, which are entity indices

         Both indices refer to those used by the news_feature_file and the
         loaded word/entity embeddings.
        """
        doc_size = model.hparams.doc_size
        batch_size = model.hparams.batch_size

        inputs = request.get_json()
        cni = np.array([inputs['words']])
        cnei = np.array([inputs['entities']])

        batch = model.iterator.gen_infer_feed_dict({
            'candidate_news_index_batch': np.pad(cni, ((0, batch_size-1), (0,0)), 'constant', constant_values=(0)),
            'candidate_news_entity_index_batch': np.pad(cnei, ((0, batch_size-1), (0,0)), 'constant', constant_values=(0)),
        })

        emb = model.infer_embedding(model.sess, batch)[0][0]
        return jsonify(emb.tolist())

    return app
