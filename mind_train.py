import sys
import os
from tempfile import TemporaryDirectory
import logging
import json
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from process_mind import (maybe_download, 
                            download_mind, 
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
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    
    # DKN parameters
    epochs = 30
    history_size = 50
    batch_size = 100

    # Mind parameters - options: demo, small, large
    MIND_SIZE = "small"

    # DKN parameters
    epochs = 10
    history_size = 50
    batch_size = 100

    # Temp dir
    tmpdir = TemporaryDirectory()

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
    print("-------------------------------------------------------")
    print("Building DKN ...")
    print("-------------------------------------------------------")
    strategy = tf.distribute.MirroredStrategy(gpus)
    print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        # TODO: convert training_file to tf.data.Dataset object and use here.
        model = DKN(hparams, DKNTextIterator)

    # model = DKN(hparams, DKNTextIterator)

    print("-------------------------------------------------------")
    print("Training DKN ...")
    print("-------------------------------------------------------")
    model.fit(train_file, valid_file)

    # Evaluation
    print("-------------------------------------------------------")
    print("Evaluating DKN ...")
    print("-------------------------------------------------------")
    res = model.run_eval(valid_file)
    print(res)

    # Saving model
    print("-------------------------------------------------------")
    print("Saving DKN ...")
    print("-------------------------------------------------------")
    model_path = os.path.join(data_path, "model")
    os.makedirs(model_path, exist_ok=True)

    model.model.save_weights(os.path.join(model_path, "dkn_ckpt"))

if __name__ == "__main__":
    main()