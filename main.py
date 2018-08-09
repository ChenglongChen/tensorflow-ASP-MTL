
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from core.data import DataLoader
from core.model import SharedPrivateModel
from core.vocab import Vocabulary
from utils import os_utils


DATA_DIR = "./data/mtl-dataset"
TASK_NAMES = [
      "apparel",
      "baby",
      "books",
      "camera_photo",
      # "dvd",
      "electronics",
      "health_personal_care",
      "imdb",
      "kitchen_housewares",
      "magazines",
      # "MR",
      "music",
      "software",
      "sports_outdoors",
      "toys_games",
      "video",
]


params = {

    "model_name": "shared_private_model",


    # dir
    "model_dir": "./model",
    "summary_dir": "./summary",


    # sequence
    "max_num_word": -1,
    "max_seq_len_word": 200,
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",


    # embedding
    "embedding_dropout": 0.5,
    "embedding_dim": 50,
    "embedding_trainable": True,
    "embedding_mask_zero": False,


    # optimization
    "training_mode": "joint",  # "sequential"

    "loss_adv_weight": 0.05,
    "loss_diff_weight": 0.01,
    "loss_l2_lambda": 0.,

    "batch_size": 16,
    "max_batch": 10000,

    "optimizer_type": "lazyadam",
    "init_lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "decay_steps": 2000,
    "decay_rate": 0.95,
    "schedule_decay": 0.004,
    "random_seed": 2018,
    "eval_every_num_update": 100,


    # encoder feature layer
    "encode_method": "textcnn",
    "attend_method": ["max"],
    "attention_dim": 64,
    "attention_num_heads": 1,

    # cnn
    "cnn_num_layers": 1,
    "cnn_num_filters": 64,
    "cnn_filter_sizes": [3, 4, 5],
    "cnn_timedistributed": False,
    "cnn_activation": tf.nn.relu,
    "cnn_gated_conv": False,
    "cnn_residual": False,


    # final layer
    "fc_dropout": 0.5,

}


def main():

    os_utils._makedirs(params["summary_dir"], force=True)
    X_train, y_train, X_test, y_test = DataLoader.load(DATA_DIR, TASK_NAMES)

    vocabulary = Vocabulary(max_num_word=params["max_num_word"])
    words = []
    for val in X_train.values():
        words.extend(val)
    vocabulary.fit(words)
    params["max_num_word"] = vocabulary.max_num_word
    for task_name in TASK_NAMES:
        X_train[task_name] = vocabulary.transform(X_train[task_name])
        X_test[task_name] = vocabulary.transform(X_test[task_name])

        X_train[task_name] = pad_sequences(X_train[task_name],
                                           maxlen=params["max_seq_len_word"],
                                           padding=params["pad_sequences_padding"],
                                           truncating=params["pad_sequences_truncating"])
        X_test[task_name] = pad_sequences(X_test[task_name],
                                          maxlen=params["max_seq_len_word"],
                                          padding=params["pad_sequences_padding"],
                                          truncating=params["pad_sequences_truncating"])

    model = SharedPrivateModel(params, TASK_NAMES)
    model.fit(X_train, y_train, X_valid=X_test, y_valid=y_test)


if __name__ == '__main__':
    main()
