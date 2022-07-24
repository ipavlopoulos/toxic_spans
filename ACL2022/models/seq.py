import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from scipy.stats import sem
from ast import literal_eval
import tensorflow 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_text as tf_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LayerNormalization
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC 
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError 
import os
from transformers import BertTokenizer, TFBertModel, BertConfig, PreTrainedTokenizerFast
from transformers import BertTokenizerFast,  BatchEncoding
from tokenizers import Encoding



class BILSTM_SEQ():
  def __init__(self,
               rnn_layer_sizes = [128],
               layer_normalize = [True],
               dropouts = [0.1],
               show_summary=True,
               patience=3,
               epochs=1000,
               batch_size=128,
               lr=0.001,
               loss='MSE',
               max_seq_len = 128,
               embedding_size = 200,
               monitor_loss = 'val_loss',
               metrics = [MeanSquaredError(name='MSE'),
                          MeanAbsoluteError(name='MAE'),
                          MeanSquaredLogarithmicError(name='MSLE'),
                        ]
              ):
        self.lr = lr
        self.batch_size = batch_size
        self.rnn_layer_sizes = rnn_layer_sizes
        self.layer_normalize = layer_normalize
        self.dropouts = dropouts
        self.max_seq_len =  max_seq_len
        self.show_summary = show_summary
        self.patience=patience
        self.epochs = epochs
        self.loss = loss
        self.embedding_size = embedding_size
        self.monitor_loss = monitor_loss
        self.metrics = metrics
        self.earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor=self.monitor_loss,
                                                          patience=self.patience,
                                                          verbose=1,
                                                          restore_best_weights=True,
                                                          mode='min'
                                                        )
        self.unk_token = '[unk]'
        self.pad_token = '[pad]'


  def build(self, vocab_size):
    """
    This method builds the model (defining the model's architecture).
    :param vocab_size: the size of the model's vocab.
    """

    inputs= Input(shape=(self.max_seq_len), name='inputs')
    x = inputs
    x = Embedding(input_dim=vocab_size, output_dim=self.embedding_size, 
                  input_length=self.max_seq_len, mask_zero=True, trainable=True)(inputs)
    for i in range(len(self.rnn_layer_sizes)):
      if self.dropouts[i]:
        x = Dropout(self.dropouts[i])(x)
      x = Bidirectional(LSTM(self.rnn_layer_sizes[i], return_sequences=True))(x)
      if self.layer_normalize[i]:
        x = LayerNormalization()(x)
    pred = TimeDistributed(Dense(1, activation=None))(x)
    pred = tensorflow.squeeze(pred)
    self.model = Model(inputs=inputs, outputs=pred)
    self.model.compile(loss=self.loss,
                      optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=self.metrics)
    if self.show_summary:
     self.model.summary()

  def create_vocab(self, tokenized_texts):
    """
    This method creates the vocab of the model. 
    :param tokenized_texts: the tokenized texts from which the vocab will be created. 
    """

    self.vocab = {w for txt in tokenized_texts for w in txt}
    self.vocab_size = len(self.vocab) + 2
    print('Vocab size: ', self.vocab_size)
    self.w2i = {w: i+2 for i,w in enumerate(self.vocab)}
    self.w2i[self.unk_token] = 1
    self.w2i[self.pad_token] = 0
    self.i2w = {i+2: self.w2i[w] for i,w in enumerate(self.vocab)}
    self.i2w[1] = self.unk_token
    self.i2w[0] = self.pad_token
  
  def to_sequences(self, tokenized_texts):
    """
    This method transforms each tokenized text to a sequence of integers.
    :param tokenized_texts: the tokenized texts which will be transformed to a sequence of integers.
    :return: the sequences of integers
    """

    #For each word of each text in tokenized texts, check if this word exists in the vocab
    #If it exits then take its index, otherwise take the index of the unknown token
    x = [[self.w2i[w] if w in self.w2i else self.w2i[self.unk_token] for w in t] for t in tokenized_texts]
    x = pad_sequences(sequences=x, maxlen=self.max_seq_len, padding='post', value=0)  # padding
    return x
  
  def fit(self, tokenized_texts, token_labels, val_tokenized_texts, val_token_labels):
    """
    This method fits the model.
    :param tokenized_texts: the tokenized train texts on which the model will be trained.
    :param token_labels: the toxicity labels of the tokens.
    :param val_tokenized_texts: the tokenized val texts on which the model will be trained.
    :param val_token_labels: the validation toxicity labels of the tokens.
    :return: the history of the model's training
    """

    # Create vocab and lookup tables
    self.create_vocab(tokenized_texts)
    # turn the tokenized texts and token labels to padded sequences of indices
    X = self.to_sequences(tokenized_texts)
    y = pad_sequences(maxlen=self.max_seq_len, sequences=token_labels, padding='post', value=0.0, dtype='float32')
    # build the model and compile it
    self.build(self.vocab_size)
    # start training
    vx = self.to_sequences(val_tokenized_texts)
    vy = pad_sequences(maxlen=self.max_seq_len, sequences=val_token_labels, padding='post', value=0.0, dtype='float32')
    history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(vx, vy), verbose=1, callbacks=[self.earlystop])
    return history
  
  def predict(self, tokenized_texts):
        predictions = self.model.predict(self.to_sequences(tokenized_texts))
        return [p.flatten() for p in predictions]

  #label the tokens of each sequence as toxic or not 
  def get_toxic_offsets(self, tokenized_texts, threshold=0.5):
    """
    This method labels the tokens of each sequence as toxic or not.
    :param tokenized_texts: the tokenized texts from which the toxic spans will be extracted.
    :param threshold: the attention threshold. If a token's attention is higher than this threshold, then the token is labeled as toxic.
    :return: the toxicity labels of each token in the text.
    """
    text_predictions = self.predict(tokenized_texts)
    output = []
    for tokens, scores in list(zip(tokenized_texts, text_predictions)):
      start = 0
      end = min(len(tokens),self.max_seq_len) #ignore pad tokens
      decisions = [1 if scores[i]>threshold else 0 for i in range(start, end)] #1 if token was found toxic by the classifier else 0
      output.append(decisions)
    return output

  def get_toxic_char_offsets(self, token_offsets, toxic_offsets):
    """
    This method extracts the char offsets of the tokens that are labeled as toxic. 
    param: token_offsets: a list that contains lists, where each list contains the character offsets of each token (of a sequence).
    param: toxic offsets: a list that contains lists, where each list contains the labels of all tokens (toxic or not) of a sequence.
    return: a list containing lists, where each list contains the char offsets of the toxic spans that were found for a given sequence.
    """
    
    toxic_char_offsets = []
    for i,instance in enumerate(toxic_offsets): #for each sequence
      instance_toxic_char_offsets = []
      for j,token_label in enumerate(instance): #for each token of the sequence 
        if token_label == 1: #check if this token was found as toxic by the classifier 
          instance_toxic_char_offsets.extend([ch for ch in token_offsets[i][j]]) #add the char offsets of this token to the list of chars returned by the model 
      toxic_char_offsets.append(instance_toxic_char_offsets)
    return toxic_char_offsets #for each sequence return a list containing the char offsets of toxic spans
    

class BERT_SEQ():

  def __init__(self,
                 max_seq_length=128,
                 show_summary=False,
                 patience=3,
                 epochs=100,
                 batch_size=32,
                 lr=2e-05,
                 session=None,
                 dense_activation = 'sigmoid',
                 loss='binary_crossentropy',
                 monitor_loss = 'val_loss',
                 monitor_mode = 'min',
                 METRICS = [BinaryAccuracy(name='accuracy'),
                        Precision(name='precision'),
                        Recall(name='recall'),
                        AUC(name='auc')]     
                 ):
        self.session = session
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False,  max_length=max_seq_length,pad_to_max_length=True)
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.show_summary = show_summary
        self.patience=patience
        self.epochs = epochs
        self.METRICS = METRICS
        self.loss = loss
        self.monitor_loss = monitor_loss
        self.monitor_mode = monitor_mode
        self.dense_activation = dense_activation
        self.earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor=self.monitor_loss,
                                                            patience=self.patience,
                                                            verbose=1,
                                                            restore_best_weights=True,
                                                            mode=self.monitor_mode)
        self.BERT = TFBertModel.from_pretrained('bert-base-cased', output_attentions = True) #, config=self.bert_config)


  def extract_xy(self, data, tokenizer):
    """
    This method aligns x and y according to BERT's sub-tokens
    :param data: the dataframe (read toxic_spans.csv)
    :param tokenizer: bert's tokenizer
    :return: x and y aligned (subtokens aligned with subtokens toxicity labels)
    """
    
    x = [] #tokens (or subtokens)
    y = [] #token labels
    for i in tqdm(range(data.shape[0])):
      subtokens = []
      token_labels = []
      tokenized_batch : BatchEncoding = tokenizer(data.iloc[i].text_of_post)
      tokenized_text :Encoding = tokenized_batch[0]
      tokens = ['[CLS]'] + tokenizer.tokenize(data.iloc[i].text_of_post) + ['[SEP]']
      for j,token in enumerate(tokens):
        if j == 0 or j == len(tokens) - 1: #ignore ['CLS'] and ['SEP'] tokens
         continue
        else:
          (start, end) = tokenized_text.token_to_chars(j) #char offset of jth sub-token (in the original text)
          span_score = []
          for ch_offset in range(start,end):
            if ch_offset in data.iloc[i].position_probability.keys():
              span_score.append(data.iloc[i].position_probability[ch_offset])
            else:
              span_score.append(0)
          token_labels.append(np.mean(span_score))
          subtokens.append(token)
      x.append(subtokens)
      y.append(token_labels)
    return x, y

  def to_bert_input(self, texts):
    """ 
    This method returns the inputs that Bert needs: input_ids, input_masks, input_segments 
    :param texts: the texts that will be converted into bert inputs 
    :return: a tuple containing the input_ids, input_masks, input_segments 
    """
    input_ids, input_masks, input_segments = [],[],[]
    for text in tqdm(texts):
      inputs = self.tokenizer.encode_plus(text, add_special_tokens=False, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                  return_attention_mask=True, return_token_type_ids=True)
      input_ids.append(inputs['input_ids'])
      input_masks.append(inputs['attention_mask'])
      input_segments.append(inputs['token_type_ids'])
    return (np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32'))

  def build(self):
    """
    This method builds the model (defining the model's architecture).
    """

    #encode the text via BERT
    in_id = Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
    in_mask = Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
    in_segment = Input(shape=(self.max_seq_length,), name="segment_ids", dtype='int32')
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = self.BERT(bert_inputs).last_hidden_state
    bert_output = bert_output[:,:,:] #remember tokenizer ignores special tokens

    x = tensorflow.keras.layers.BatchNormalization()(bert_output)
    pred = TimeDistributed(Dense(1, activation=self.dense_activation))(x)
    self.model = Model(inputs=bert_inputs, outputs=pred)
    self.model.compile(loss=self.loss,
                      optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=self.METRICS)
    if self.show_summary:
      self.model.summary()

  def fit(self, train_X, train_y, val_X, val_y):
    """
    This method fits the model.
    :param train_X: the train texts on which the model will be trained.
    :param train_y: the toxicity labels of the tokens of the train texts.
    :param val_X: the val texts on which the model will be evaluated.
    :param val_y: the toxicity labels of the the tokens of the val texts.
    :return: the history of the model's training
    """

    train_input = self.to_bert_input(train_X)
    val_input = self.to_bert_input(val_X)
    train_y = pad_sequences(maxlen=self.max_seq_length, sequences=train_y, padding='post', value=0.0, dtype='float32')
    val_y = pad_sequences(maxlen=self.max_seq_length, sequences=val_y, padding='post', value=0.0, dtype='float32')
    self.build()
    history = self.model.fit(train_input,
                             train_y,
                             validation_data=(val_input, val_y),
                             epochs=self.epochs,
                             callbacks=[self.earlystop],
                             batch_size=self.batch_size,
                             class_weight=None 
                            )
    return history

  def predict(self, texts):
    test_input = self.to_bert_input(texts)
    predictions = self.model.predict(test_input)
    print('Stopped epoch: ', self.earlystop.stopped_epoch)
    return [p.flatten() for p in predictions]
  
  #label the tokens of each sequence  as toxic or not 
  def get_toxic_offsets(self, texts, threshold=0.5):
    """
    This method labels the tokens of each sequence as toxic or not.
    :param texts: the texts from which the toxic spans will be extracted.
    :param threshold: the attention threshold. If a token's attention is higher than this threshold, then the token is labeled as toxic.
    :return: the toxicity labels of each token in the text.
    """

    text_predictions = self.predict(texts)
    output = []
    tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
    for tokens, scores in list(zip(tokenized_texts, text_predictions)):
      start = 0 #ignore predictions for cls token #0
      end = min(len(tokens),self.max_seq_length) #ignore pad tokens and sep token 
      decisions = [1 if scores[i]>threshold else 0 for i in range(start, end)] #1 if token was found toxic by the classifier else 0
      output.append(decisions)
    return output
 
  def get_toxic_char_offsets(self, texts, toxic_offsets):
    """
      This method extracts the char offsets of the tokens that are labeled as toxic. 
      :param texts: a list that contains the test texts.
      :param toxic offsets: a list that contains lists, where each list contains the labels of all tokens (toxic or not) of a sequence.
      :return: a list containing lists, where each list contains the char offsets of the toxic spans that were found for a given sequence.
    """

    toxic_char_offsets = []
    for i,text in enumerate(texts):
      instance_toxic_char_offsets = []
      tokenized_batch : BatchEncoding = self.tokenizer(text)
      tokenized_text :Encoding = tokenized_batch[0]
      tokens = ['CLS'] + self.tokenizer.tokenize(text) + ['SEP']
      for j,token_label in enumerate([0] + toxic_offsets[i] + [0]): #adding 2 pseudo labels for 'CLS' and 'SEP' tokens
        if j == 0 or (j == len(tokens)-1 and len(tokens) <= self.max_seq_length): #ignore ['CLS'] and ['SEP'] tokens
          continue
        if token_label == 1: #if token (or subtoken) was found toxic by the classifier
          index_of_word = tokenized_text.token_to_word(j) #get the index of the word in the sequence 
          (start, end) = tokenized_text.word_to_chars(index_of_word) #get the char offsets of the first and last chars of this token (subtoken)
          instance_toxic_char_offsets.extend([ch for ch in range(start,end)]) #add the char offsets of this token (subtoken) to the list
      toxic_char_offsets.append(set(instance_toxic_char_offsets)) #set removes the duplicates char offsets (if a subtoken was found toxic, we add the char offsets of the whole word)
    return toxic_char_offsets
  
  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)
        
