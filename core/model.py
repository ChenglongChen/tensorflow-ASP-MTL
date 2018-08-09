
import numpy as np
import tensorflow as tf

from sklearn.metrics import log_loss
from tensorflow.python.framework import ops

from tf_common.nn_module import encode, attend, word_dropout
from tf_common.optimizer import LazyNadamOptimizer


class FlipGradientBuilder(object):
    """
    Code: https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
    """
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


class SharedPrivateModel(object):

    def __init__(self, params, task_names, init_embedding_matrix=None):

        self.model_name = params["model_name"]
        self.params = params
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.task_name_mapper = dict(zip(task_names, range(self.num_tasks)))
        self.init_embedding_matrix = init_embedding_matrix

        self._build()
        self.summary = self._get_summary()
        self.sess, self.saver = self._init_session()
        self.train_writer = tf.summary.FileWriter(self.params["summary_dir"] + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.params["summary_dir"] + '/test')


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models
        saver = tf.train.Saver(max_to_keep=None)
        return sess, saver


    def save_session(self):
        self.saver.save(self.sess, self.params["weights_dir"] + "/model.checkpoint")


    def restore_session(self):
        self.saver.restore(self.sess, self.params["weights_dir"] + "/model.checkpoint")


    def _get_summary(self):
        with tf.name_scope(self.model_name + "/"):
            tf.summary.scalar("loss_all", self.all_loss)
            tf.summary.scalar("acc_all", self.all_acc)
            for task_name in self.task_names:
                tf.summary.scalar("loss_"+task_name, self.loss[task_name])
                tf.summary.scalar("acc_"+task_name, self.acc[task_name])
            # error: https://blog.csdn.net/u012436149/article/details/53894364
            # summary = tf.summary.merge_all()
            summary = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES, self.model_name)
            )
        return summary


    def _get_embedding_matrix(self):
        if self.init_embedding_matrix is None:
            std = 0.1
            minval = -std
            maxval = std
            emb_matrix = tf.Variable(
                tf.random_uniform(
                    # 0: padding
                    # max_num_word + 1: oov
                    [self.params["max_num_word"] + 2, self.params["embedding_dim"]],
                    minval, maxval,
                    seed=self.params["random_seed"],
                    dtype=tf.float32))
        else:
            emb_matrix = tf.Variable(self.init_embedding_matrix,
                                     trainable=self.params["embedding_trainable"])
        return emb_matrix


    def _base_feature_extractor(self, emb_seq, seq_len, name, reuse):
        #### encode
        input_dim = self.params["embedding_dim"]
        enc_seq = encode(emb_seq, method=self.params["encode_method"],
                         input_dim=input_dim,
                         params=self.params,
                         sequence_length=seq_len,
                         mask_zero=self.params["embedding_mask_zero"],
                         scope_name=self.model_name + "_encode_%s"%name, reuse=reuse,
                         training=self.training)
        #### attend
        feature_dim = self.params["encode_dim"]
        context = None
        att_seq = attend(enc_seq, context=context,
                         encode_dim=self.params["encode_dim"],
                         feature_dim=feature_dim,
                         attention_dim=self.params["attention_dim"],
                         method=self.params["attend_method"],
                         scope_name=self.model_name + "_attention_%s"%name,
                         reuse=reuse, num_heads=self.params["attention_num_heads"])
        return att_seq


    def _shared_feature_extractor(self, emb_seq, seq_len):
        with tf.name_scope("shared_part/"):
            return self._base_feature_extractor(emb_seq, seq_len, name="shared_part", reuse=tf.AUTO_REUSE)


    def _private_feature_extractor(self, emb_seq, seq_len, task_name):
        with tf.name_scope(task_name + "/"):
            return self._base_feature_extractor(emb_seq, seq_len, name="%s_private_part"%task_name, reuse=False)


    def _shared_task_discriminator(self, feature):
        with tf.name_scope("shared_part/"):
            return tf.layers.dense(feature, len(self.task_names), reuse=tf.AUTO_REUSE, name="shared_task_discriminator")


    def _adversarial_loss(self, shared_samples, task_labels):
        with tf.name_scope("shared_part/"):
            shared_samples = self.flip_gradient(shared_samples)
            shared_samples = tf.layers.Dropout(self.params["fc_dropout"])(shared_samples, self.training)

            logits = self._shared_task_discriminator(shared_samples)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=task_labels, logits=logits)
            loss = tf.reduce_mean(loss)

            return loss


    def _difference_loss(self, private_samples, shared_samples, weight=1.0):
        '''
        Paper: Domain Separation Networks
        Code: https://github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/losses.py
        '''
        with tf.name_scope("shared_part/"):
            private_samples -= tf.reduce_mean(private_samples, 0)
            shared_samples -= tf.reduce_mean(shared_samples, 0)

            private_samples = tf.nn.l2_normalize(private_samples, 1)
            shared_samples = tf.nn.l2_normalize(shared_samples, 1)

            correlation_matrix = tf.matmul(
                private_samples, shared_samples, transpose_a=True)

            cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
            cost = tf.where(cost > 0, cost, 0, name='value')

            assert_op = tf.Assert(tf.is_finite(cost), [cost])
            with tf.control_dependencies([assert_op]):
                loss = tf.identity(cost)

            return loss


    def _build_task_graph(self, task_name):

        #### tf vars
        self.task_labels[task_name] = tf.placeholder(tf.int32, shape=[None], name="task_labels")
        self.labels[task_name] = tf.placeholder(tf.int32, shape=[None], name="labels")
        self.seq_word[task_name] = tf.placeholder(tf.int32, shape=[None, None], name="seq_word")

        #### embedding
        emb_seq = tf.nn.embedding_lookup(self.emb_matrix, self.seq_word[task_name])
        emb_seq = word_dropout(emb_seq,
                               training=self.training,
                               dropout=self.params["embedding_dropout"],
                               seed=self.params["random_seed"])

        #### features
        shared_features = self._shared_feature_extractor(emb_seq, seq_len=None)
        private_features = self._private_feature_extractor(emb_seq, seq_len=None, task_name=task_name)

        feature = tf.concat([shared_features, private_features], axis=1)
        feature = tf.layers.Dropout(self.params["fc_dropout"])(feature, self.training)

        #### task classifier
        logits = tf.layers.dense(feature, 2)
        probas = tf.nn.softmax(logits)
        loss_task = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels[task_name], logits=logits)
        loss_task = tf.reduce_mean(loss_task)

        #### shared classifier
        loss_adv = self._adversarial_loss(shared_features, self.task_labels[task_name])
        loss_diff = self._difference_loss(shared_features, private_features)

        #### loss
        loss = loss_task
        if self.params["loss_adv_weight"] > 0:
            loss += self.params["loss_adv_weight"] * loss_adv
        if self.params["loss_diff_weight"] > 0:
            loss += self.params["loss_diff_weight"] * loss_diff
        if self.params["loss_l2_lambda"] > 0:
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name])
            loss += self.params["loss_l2_lambda"] * l2_losses

        #### accuracy
        preds = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        acc = tf.cast(tf.equal(preds, self.labels[task_name]), tf.float32)
        acc = tf.reduce_mean(acc)

        return probas, loss, acc


    def _build(self):
        self.task_labels = {}
        self.labels = {}
        self.seq_word = {}
        self.train_ops = {}
        self.probas = {}
        self.loss = {}
        self.acc = {}
        self.training = tf.placeholder(tf.bool, shape=[], name="training")
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                        self.params["decay_steps"], self.params["decay_rate"])
        self.emb_matrix = self._get_embedding_matrix()
        self.flip_gradient = FlipGradientBuilder()
        optimizer = {}
        for task_name in self.task_names:
            self.probas[task_name], self.loss[task_name], self.acc[task_name] = self._build_task_graph(task_name)
            with tf.variable_scope("optimizer_" + task_name, reuse=tf.AUTO_REUSE):
                optimizer[task_name] = LazyNadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                          beta2=self.params["beta2"], epsilon=1e-8,
                                                          schedule_decay=self.params["schedule_decay"])
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_ops[task_name] = optimizer[task_name].minimize(self.loss[task_name], global_step=self.global_step)
        self.all_loss = tf.add_n(list(self.loss.values()))/self.num_tasks
        self.all_acc = tf.add_n(list(self.acc.values()))/self.num_tasks
        self.all_train_ops = list(self.train_ops.values())


    def _get_feed_dict_for_train(self, X_train, y_train):
        feed_dict = {}
        for task_name in X_train.keys():
            l = len(y_train[task_name])
            idx = np.random.choice(range(l), self.params["batch_size"])
            feed_dict.update({
                self.task_labels[task_name]: self.task_name_mapper[task_name]*np.ones(len(idx)),
                self.labels[task_name]: y_train[task_name][idx],
                self.seq_word[task_name]: X_train[task_name][idx],
                self.training: True,
            })
        return feed_dict


    def _get_feed_dict_for_infer(self, X, idx, task_name):
        feed_dict = {
            self.seq_word[task_name]: X[idx],
            self.training: False,
        }
        return feed_dict


    def _get_batch_index(self, seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        # last batch
        if len(res) * step < n:
            res.append(seq[len(res) * step:])
        return res


    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        total_loss = 0.
        total_acc = 0.
        metric_decay = 0.9
        for batch in range(self.params["max_batch"]):
            feed_dict = self._get_feed_dict_for_train(X_train, y_train)
            if self.params["training_mode"] == "joint":
                op = []
                op.extend(self.all_train_ops)
                op.append(self.summary)
                op.append(self.all_loss)
                op.append(self.all_acc)
                output = self.sess.run(op, feed_dict)
                summary = output[-3]
                all_loss = output[-2]
                all_acc = output[-1]
            elif self.params["training_mode"] == "sequential":
                all_loss = 0.
                all_acc = 0.
                for task_name in self.task_names:
                    op = [self.train_ops[task_name], self.summary, self.loss[task_name], self.acc[task_name]]
                    _, summary, loss, acc = self.sess.run(op, feed_dict)
                    all_loss += loss
                    all_acc += acc
                all_loss /= float(self.num_tasks)
                all_acc /= float(self.num_tasks)

            total_loss = metric_decay * total_loss + (1. - metric_decay) * all_loss
            total_acc = metric_decay * total_acc + (1. - metric_decay) * all_acc
            if (X_valid is not None) and \
                    (y_valid is not None) and \
                    ((batch + 1) % self.params["eval_every_num_update"] == 0):
                summary, total_loss_valid, total_acc_valid = self.evaluate(X_valid, y_valid)
                self.test_writer.add_summary(summary, batch + 1)
            else:
                self.train_writer.add_summary(summary, batch + 1)


    def _predict_proba_inner(self, X, task_name):
        l = X.shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        y = []
        y_append = y.append
        for idx in batches:
            feed_dict = self._get_feed_dict_for_infer(X, idx, task_name)
            p = self.sess.run(self.probas[task_name], feed_dict=feed_dict)
            y_append(p)
        y = np.vstack(y).astype(np.float32)
        return y


    def predict_proba(self, X):
        y = {}
        for task_name in X.keys():
            p = self._predict_proba_inner(X[task_name], task_name)
            y[task_name] = p[:,1]
        return y


    def predict(self, X):
        y = {}
        for task_name in X.keys():
            p = self._predict_proba_inner(X[task_name], task_name)
            y[task_name] = np.argmax(p, 1)
        return y


    def evaluate(self, X, y):
        all_loss = 0.
        all_acc = 0.
        summary = tf.Summary()
        for task_name in y.keys():
            p = self._predict_proba_inner(X[task_name], task_name)
            y_pred = np.argmax(p, 1)
            loss = log_loss(y_true=y[task_name], y_pred=p[:,1])
            acc = np.mean(y[task_name] == y_pred)
            all_loss += loss
            all_acc += acc
            summary.value.add(tag="loss_"+task_name, simple_value=loss)
            summary.value.add(tag="acc_"+task_name, simple_value=acc)
        all_loss /= float(len(y.keys()))
        all_acc /= float(len(y.keys()))
        summary.value.add(tag="loss_all", simple_value=all_loss)
        summary.value.add(tag="acc_all", simple_value=all_acc)
        return summary, all_loss, all_acc
