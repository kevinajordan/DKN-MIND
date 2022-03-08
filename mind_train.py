import sys
import os
from tempfile import TemporaryDirectory
import logging
import json
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from deeprec_utils import download_deeprec_resources, prepare_hparams
from dkn import DKN
from dkn_iterator import DKNTextIterator


def main():
    print(f"System version: {sys.version}")
    print(f"Tensorflow version: {tf.__version__}")
    tmpdir = TemporaryDirectory()
    MIND_SIZE = "small"
    # DKN parameters
    epochs = 30
    history_size = 50
    batch_size = 100

    # Download and load MIND demo data
    data_path = os.path.join(tmpdir.name, "mind-demo-dkn")

    yaml_file = os.path.join(data_path, r'dkn.yaml')
    train_file = os.path.join(data_path, r'train_mind_demo.txt')
    valid_file = os.path.join(data_path, r'valid_mind_demo.txt')
    test_file = os.path.join(data_path, r'test_mind_demo.txt')
    news_feature_file = os.path.join(data_path, r'doc_feature.txt')
    user_history_file = os.path.join(data_path, r'user_history.txt')
    wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
    entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')
    contextEmb_file = os.path.join(data_path, r'TransE_context2vec_100.npy')
    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/deeprec/', tmpdir.name, 'mind-demo-dkn.zip')
    '''
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
    '''
    # Create hyperparameters
    print("Creating hyperparameters...")
    hparams = prepare_hparams(yaml_file,
                            news_feature_file = news_feature_file,
                            user_history_file = user_history_file,
                            wordEmb_file=wordEmb_file,
                            entityEmb_file=entityEmb_file,
                            contextEmb_file=contextEmb_file,
                            epochs=epochs,
                            history_size=history_size,
                            batch_size=batch_size)
    print(hparams)

    # Train DKN
    print("Building DKN...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DKN(hparams, DKNTextIterator)

    #model = DKN(hparams, DKNTextIterator)

    print("Training DKN...")
    model.fit(train_file, valid_file)

    # Evaluation
    print("Evaluating DKN...")
    res = model.run_eval(valid_file)
    print(res)

if __name__ == "__main__":
    main()