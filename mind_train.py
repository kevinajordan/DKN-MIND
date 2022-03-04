import sys

import os
from tempfile import TemporaryDirectory
import logging
import papermill as pm
import scrapbook as sb
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from process_mind  import try_download
from process_mind import (download_mind, 
                                     extract_mind, 
                                     read_clickhistory, 
                                     get_train_input, 
                                     get_valid_input, 
                                     get_user_history,
                                     get_words_and_entities,
                                     generate_embeddings) 
from deeprec_utils import prepare_hparams
from dkn import DKN
from dkn_iterator import DKNTextIterator


def main():
    print(f"System version: {sys.version}")
    print(f"Tensorflow version: {tf.__version__}")
    tmpdir = TemporaryDirectory()
    MIND_SIZE = "small"
    # DKN parameters
    epochs = 10
    history_size = 50
    batch_size = 100

    # Paths
    data_path = os.path.join(tmpdir.name, "mind-dkn")
    train_file = os.path.join(data_path, "train_mind.txt")
    valid_file = os.path.join(data_path, "valid_mind.txt")
    user_history_file = os.path.join(data_path, "user_history.txt")
    infer_embedding_file = os.path.join(data_path, "infer_embedding.txt")

    # Data preparation
    print("Starting data preparation...")
    train_zip, valid_zip = download_mind(size=MIND_SIZE, dest_path=data_path)
    train_path, valid_path = extract_mind(train_zip, valid_zip)
    print ("Reading click history...")
    train_session, train_history = read_clickhistory(train_path, "behaviors.tsv")
    valid_session, valid_history = read_clickhistory(valid_path, "behaviors.tsv")
    print ("Generating training and validation files...")
    get_train_input(train_session, train_file)
    get_valid_input(valid_session, valid_file)
    print("Getting user history...")
    get_user_history(train_history, valid_history, user_history_file)
    train_news = os.path.join(train_path, "news.tsv")
    valid_news = os.path.join(valid_path, "news.tsv")
    print("Generating embeddings...")
    news_words, news_entities = get_words_and_entities(train_news, valid_news)
    train_entities = os.path.join(train_path, "entity_embedding.vec")
    valid_entities = os.path.join(valid_path, "entity_embedding.vec")
    news_feature_file, word_embeddings_file, entity_embeddings_file = generate_embeddings(
        data_path,
        news_words,
        news_entities,
        train_entities,
        valid_entities,
        max_sentence=10,
        word_embedding_dim=100,
    )

    # Create hyperparameters
    print("Creating hyperparameters...")
    yaml_file = maybe_download(url="https://recodatasets.z20.web.core.windows.net/deeprec/deeprec/dkn/dkn_MINDsmall.yaml", 
                           work_directory=data_path)
    hparams = prepare_hparams(yaml_file,
                            news_feature_file=news_feature_file,
                            user_history_file=user_history_file,
                            wordEmb_file=word_embeddings_file,
                            entityEmb_file=entity_embeddings_file,
                            epochs=epochs,
                            history_size=history_size,
                            batch_size=batch_size)

    # Train DKN
    print("Building DKN...")
    model = DKN(hparams, DKNTextIterator)

    print("Training DKN...")
    model.fit(train_file, valid_file)

    # Evaluation
    print("Evaluating DKN...")
    res = model.run_eval(valid_file)
    print(res)

if __name__ == "__main__":
    main()