import random
random.seed(a=2021)
import lime
import pandas as pd
import numpy as np
np.random.seed(seed=2021)
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GRU, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def write_offsets(offsets, filename="answer.txt"):
    """
    Static method to write out the offsets of the toxic spans.
    :param offsets: list of (lists of) offsets
    :param filename: the name of the file to write to
    :return:
    """
    with open(filename) as o:
        for offsets in offsets:
            o.write(offsets + "\n")


class Random:

    def __init__(self, texts):
        """
        Random baseline for toxic span detection.
        Returns toxic offsets at random.
        :param text:
        """
        self.toxic_offsets = [[i for i, ch in enumerate(text) if random.random()>0.5] for text in texts]

    def get_toxic_offsets(self):
        return self.toxic_offsets


class InputErasure:

    def __init__(self,
                 classifier,
                 text,
                 one_by_one=False,
                 tokenise=lambda txt: txt.split(),
                 class_names=[0, 1],
                 mask=u"[mask]",
                 threshold=0.2,
                 reshape_predictions=True):
        """
        Given a classifier and a tokenisation method InputErasure returns the toxic words and respective offsets.
        This implementation is based on the paper "Understanding Neural Networks through Representation Erasure" by
        Li et al.

        :param classifier: any toxicity classifier that predicts a text as toxic or not
        :param text: the textual input (sentence or document) as a string
        :param one_by_one: some classifiers may require one by one classification when scoring the "ablated" texts.
        :param tokenise: by default splits the words on empty space
        :param class_names: by default "toxic" is represented by 1 and "civil" by 0
        :param mask: the pseudo token to mask the toxic word (for visualisation purposes)
        :param threshold: above this value the text is predicted toxic (default 0.2)
        :param reshape_predictions: flattens the output, some classifiers may required this to be set to False
        """
        self.class_names = class_names
        self.classifier = classifier
        self.mask = mask
        self.one_by_one = one_by_one
        self.reshape_predictions = reshape_predictions
        self.initial_score = self.clf_predict([text])
        self.tokenise = tokenise
        self.words = self.tokenise(text)
        self.ablations, self.indices = self.create_ablations()
        self.scores = self.clf_predict(self.ablations)
        self.e = 10e-05
        self.scores_decrease = [(self.initial_score - s) / (self.initial_score+self.e) for s in self.scores]
        self.threshold = threshold
        self.black_list = self.get_black_list()

    def clf_predict(self, texts):
        if self.one_by_one:
            predictions = [self.classifier.predict([t])[0] for t in texts]
        else:
            predictions = self.classifier.predict(texts)
        if self.reshape_predictions:
            predictions = predictions.reshape(1, -1)[0]
        if len(texts) == 1:
            return predictions[0]
        return predictions

    def create_ablations(self):
        ablations, indices = [], []
        for i, w in enumerate(self.words):
            words_copy = [w for w in self.words]
            words_copy[i] = self.mask
            ablations.append(" ".join(words_copy))
            indices.append(i)
        return ablations, indices

    def get_black_list(self):
        return [self.indices[i] for i, s in enumerate(self.scores_decrease) if s > self.threshold]

    def get_toxic_offsets(self):
        """
        Get the offsets of the toxic spans
        WARNING: Valid only for empty space tokenisation.
        :return: a list with offsets found toxic
        """
        current_offset = 0
        toxic_offsets = []
        for i, word in enumerate(self.words):
            if i in set(self.black_list):
                toxic_offsets.extend(list(range(current_offset, current_offset+len(word))))
            current_offset += len(word) + 1
        return toxic_offsets

    def get_mitigated_text(self):
        return " ".join([w if i not in set(self.black_list) else self.mask for i, w in enumerate(self.words)])

    def get_as_pandas(self):
        scores_pd = pd.DataFrame({"word": self.words, "indices": self.indices, "score_dec": self.scores_decrease})
        scores_pd = scores_pd.sort_values(by=["score_dec"])
        return scores_pd


class LimeUsd(InputErasure):

    def __init__(self,
                 classifier,
                 text,
                 one_by_one=False,
                 tokenise=lambda txt: txt.split(),
                 class_names=[0, 1],
                 mask=u"[mask]",
                 threshold=0.2,
                 reshape_predictions=True):
        """
        Given a classifier and a tokenisation method LimeUsd returns the toxic words and the respective offsets.
        This implementation is based on LIME.
        :param classifier: any toxicity classifier that predicts a text as toxic or not
        :param text: the textual input (sentence or document) as a string
        :param one_by_one: some classifiers may require one by one classification when scoring the "ablated" texts.
        :param tokenise: by default splits the words on empty space -- same as LIME
        :param class_names: by default "toxic" is represented by 1 and "civil" by 0
        :param mask: the pseudo token to mask the toxic word (for visualisation purposes)
        :param threshold: above this value the text is predicted toxic (default 0.2)
        :param reshape_predictions: flattens the output, some classifiers may required this to be set to False
        """
        self.class_names = class_names
        self.classifier = classifier
        self.mask = mask
        self.one_by_one = one_by_one
        self.reshape_predictions = reshape_predictions
        self.text = text
        self.initial_score = self.clf_predict([text])
        self.tokenise = tokenise
        self.explainer = LimeTextExplainer(class_names=self.class_names, split_expression=tokenise)
        self.words = self.tokenise(text)
        self.ablations, self.indices = self.create_ablations()
        self.scores_decrease = self.lime_explain(self.words)
        self.threshold = threshold
        self.black_list = self.get_black_list()

    def lime_explain(self, words):
        num_of_feats = len(words)
        predictor = lambda texts: np.array([[0, p] for p in self.classifier.predict(texts)])
        explain = self.explainer.explain_instance(self.text, predictor, num_features=num_of_feats)
        word2score = dict(explain.as_list())
        return [word2score[w] for w in self.words]


class RNNSL:

    def __init__(self, maxlen=128, batch_size=32, w_embed_size=200, padding="post", h_embed_size=200, dropout=0.1, patience=1, plot=True, max_epochs=100):
        self.maxlen = maxlen
        self.METRICS = [BinaryAccuracy(name='accuracy'),
                        Precision(name='precision'),
                        Recall(name='recall'),
                        AUC(name='auc')]
        self.w_embed_size = w_embed_size
        self.h_embed_size = h_embed_size
        self.dropout = dropout
        self.vocab_size = -1
        self.padding = padding
        self.patience = patience
        self.model = None
        self.w2i = {}
        self.epochs = max_epochs
        self.i2w = {}
        self.vocab = []
        self.batch_size = batch_size
        self.show_the_model = plot
        self.threshold = 0.2
        self.toxic_label = 2
        self.not_toxic_label = 1
        self.unk_token = "[unk]"
        self.pad_token = "[pad]"

    def build(self):
        input = Input(shape=(self.maxlen,))
        model = Embedding(input_dim=self.vocab_size+1, output_dim=self.w_embed_size, input_length=self.maxlen, mask_zero=True)(input)  # 50-dim embedding
        model = Dropout(self.dropout)(model)
        model = Bidirectional(LSTM(units=self.h_embed_size, return_sequences=True, recurrent_dropout=self.dropout))(model)  # variational biLSTM
        output = TimeDistributed(Dense(3, activation="sigmoid"))(model)
        return Model(input, output)

    def predict(self, tokenized_texts, class_num=2):
        predictions = self.model.predict(self.to_sequences(tokenized_texts))[:,:,class_num]
        return [p.flatten() for p in predictions]

    def get_toxic_offsets(self, tokenized_texts, threshold=None):
        text_predictions = self.predict(tokenized_texts)
        if threshold is None:
            threshold=self.threshold
        output = []
        for tokens, scores in list(zip(tokenized_texts, text_predictions)):
            if self.padding == "pre":
                start = self.maxlen-len(tokens) if self.maxlen>len(tokens) else 0
                end = self.maxlen
            else:
                start = 0
                end = min(len(tokens),self.maxlen)
            decisions = [self.toxic_label if scores[i]>threshold else self.not_toxic_label for i in range(start, end)]
            output.append(decisions)
        return output

    def set_up_preprocessing(self, tokenized_texts):
        self.vocab = list(set([w for txt in tokenized_texts for w in txt]))
        self.vocab_size = len(self.vocab) + 1
        self.w2i = {w: i+2 for i,w in enumerate(self.vocab)}
        self.w2i[self.unk_token] = 1
        #self.w2i[self.pad_token] = 0
        self.i2w = {i+2: self.w2i[w] for i,w in enumerate(self.vocab)}
        self.i2w[1] = self.unk_token
        #self.i2w[0] = self.pad_token

    def to_sequences(self, tokenized_texts):
        x = [[self.w2i[w] if w in self.w2i else 1 for w in t] for t in tokenized_texts]
        x = pad_sequences(sequences=x, maxlen=self.maxlen, padding=self.padding, value=0)  # padding
        return x

    def fit(self, tokenized_texts, token_labels, validation_data=None, monitor="val_loss"):
        # set up the vocabulary and the related methods
        self.set_up_preprocessing(tokenized_texts)
        # turn the tokenized texts and token labels to padded sequences of indices
        x = self.to_sequences(tokenized_texts)
        y = pad_sequences(maxlen=self.maxlen, sequences=token_labels, padding=self.padding, value=0)
        # build the model and compile it
        self.model = self.build()
        if self.show_the_model:
            print(self.model.summary())
            plot_model(self.model, show_shapes=True, to_file="neural_sequence_labeler.model.png")
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=self.METRICS)
        #mode = "max" if monitor == "val_accuracy" else "min"
        early = EarlyStopping(monitor=monitor, mode="min" if "loss" in monitor else "max", patience=self.patience, verbose=1, min_delta=0.0001, restore_best_weights=True)
        # start training
        if validation_data is not None:
            assert len(validation_data) == 2
            vx = self.to_sequences(validation_data[0])
            vy = pad_sequences(maxlen=self.maxlen, sequences=validation_data[1], padding=self.padding, value=0)
            history = self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(vx, vy), verbose=1, callbacks=[early])
        else:
            history = self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1, verbose=1, callbacks=[early])
        return pd.DataFrame(history.history)

    def tune_threshold(self, validation_data, evaluator):
        assert len(validation_data) == 2 and self.model is not None
        predictions = self.predict(validation_data[0])
        decisions = [[self.toxic_label if scores[i] > self.threshold else self.not_toxic_label for i in range(min(len(tokens), self.maxlen))] for tokens, scores in list(zip(validation_data[0], predictions))]
        opt_score = np.mean([evaluator(p, g) for p,g in list(zip(decisions,validation_data[1]))])
        for thr in range(0, 100, 1):
            decisions = [[self.toxic_label if scores[i] > thr/100. else self.not_toxic_label for i in range(min(len(tokens), self.maxlen))] for
                         tokens, scores in list(zip(validation_data[0], predictions))]
            score = np.mean([evaluator(p, g) for p, g in list(zip(decisions, validation_data[1]))])
            if score > opt_score:
                opt_score = score
                self.threshold = thr/100.


####### Text Classifiers ##############

class RNNTC():
    def __init__(self,
                 verbose=1,
                 batch_size=128,
                 n_epochs=100,
                 max_length=128,
                 word_dimension=200,
                 hidden_dimension=200,
                 loss="mse",
                 prefix=""):
        self.verbose = verbose
        self.batch_size = batch_size
        self.hidden_dimension = hidden_dimension
        self.word_dimension = word_dimension
        self.n_epochs = n_epochs
        self.max_length = max_length
        self.tokenizer = Tokenizer()
        self.loss = loss
        self.name = f'b{batch_size}.e{n_epochs}.len{max_length}.{prefix}rnn'

    def build(self, vocab_size, show=False):
        inputs1 = Input(shape=(self.max_length,))
        emb1 = Embedding(vocab_size, self.word_dimension)(inputs1)
        rnn = LSTM(self.hidden_dimension, return_sequences=False)(emb1)
        fnn = Dense(1, activation='sigmoid')(rnn)
        self.model = Model(inputs=inputs1, outputs=fnn)
        self.model.compile(loss=self.loss,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        if show:
            print(self.model.summary())
            plot_model(self.model, show_shapes=True, to_file='plot.png')

    def text_process(self, texts, tokenizer):
        x1 = tokenizer.texts_to_sequences(np.array(texts))
        x1 = sequence.pad_sequences(x1, maxlen=self.max_length)  # padding
        return x1

    def fit(self, X, y, X_val, y_val, monitor='val_loss', patience=1):
        self.tokenizer.fit_on_texts(X)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % self.vocab_size)
        self.build(self.vocab_size)
        early = EarlyStopping(monitor=monitor, patience=patience, verbose=self.verbose, min_delta=0.0001,
                              restore_best_weights=True)
        self.model.fit(self.text_process(X, self.tokenizer), y,
                       validation_data=(self.text_process(X_val, self.tokenizer), y_val),
                       epochs=self.n_epochs,
                       batch_size=self.batch_size,
                       verbose=self.verbose,
                       callbacks=[early])

    def predict(self, test_texts):
        predictions = self.model.predict(self.text_process(test_texts, self.tokenizer))
        return predictions

class MLTC:

    def __init__(self,
                 tokenise=lambda x: x.split(),
                 model=LogisticRegression(),
                 binary=True,
                 max_features=10000,
                 use_idf=False,
                 lowercase=True):
        self.tokenise = tokenise
        self.use_idf = use_idf
        self.vectorizer = CountVectorizer(analyzer='word',
                                          stop_words="english",
                                          tokenizer=tokenise,
                                          lowercase=lowercase,
                                          binary=binary,
                                          max_features=max_features)
        self.model = model

    def fit(self, X_train, y_train, X_dev=None, y_dev=None):
        train_vectors = self.vectorizer.fit_transform(X_train)
        self.model.fit(train_vectors, y_train)

    def predict(self, x):
        test_vectors = self.vectorizer.transform(x)
        return self.model.predict_proba(test_vectors)[:, 1]
