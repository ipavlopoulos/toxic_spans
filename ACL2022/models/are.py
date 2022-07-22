import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn import metrics
from scipy.stats import sem
from ast import literal_eval
import tensorflow 
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
from transformers import BertTokenizer, TFBertModel, BertConfig, PreTrainedTokenizerFast
from transformers import BertTokenizerFast,  BatchEncoding
from tokenizers import Encoding

class BILSTM_ARE():
  """
  BILSTM with deep self attention toxicity classifier.
  To detect toxic spans, we used the attention scores. We obtain a sequence of binary decisions (toxic, non-toxic) for the tokens of the post (inherited by their
  character offsets) by using a probability threshold (tuned on development data) applied to the attention scores. 
  """
  def __init__(self,
               rnn_layer_size = 128,
               attention_hidden_layers = 2,
               show_summary=True,
               patience=3,
               epochs=200,
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
        self.rnn_layer_size = rnn_layer_size
        self.attention_hidden_layers = 2
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

  def dummy_loss(self, y_true, y_pred):
    """
    This method is necessary for keras to be able to extract the attention weights (as an output of the model).
    Thats why we return 0 (to not contribute on the loss).
    """
    return 0.0

  def build(self, vocab_size):
    """
    This method builds the model (defining the model's architecture).
    :param vocab_size: the size of the model's vocab.
    """
    inputs= Input(shape=(self.max_seq_len,), name='inputs')
    x = Embedding(input_dim=vocab_size, output_dim=self.embedding_size, 
                    input_length=self.max_seq_len, mask_zero=True, trainable=True)(inputs)
    rnn = Bidirectional(LSTM(self.rnn_layer_size, return_sequences=True))(x)
    for i in range(self.attention_hidden_layers): #-1):
      if i == 0:
        x = TimeDistributed(Dense(self.rnn_layer_size, activation = None))(rnn)
      elif i == self.attention_hidden_layers-1: #last layer
        x = TimeDistributed(Dense(1, activation = None))(x)
      else:
        x = TimeDistributed(Dense(self.rnn_layer_size, activation = None))(x) #hidden layer
    att_weights = tensorflow.nn.softmax(x, axis=1)
    weighted_representation = tensorflow.math.multiply(att_weights, rnn, name=None)
    weighted_representation = tensorflow.reduce_sum(weighted_representation, axis = 1)
    pred = Dense(1, activation = None, name = 'classification')(weighted_representation)
    self.model = Model(inputs=inputs, outputs=[pred, tensorflow.squeeze(att_weights)])
    self.model.compile(loss=[self.loss, self.dummy_loss] ,
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
  
  def fit(self, tokenized_texts, y, dummy_y, val_texts, val_y, val_dummy_y):
    """
    This method fits the model.
    :param tokenized_texts: the tokenized train texts on which the model will be trained.
    :param y: the toxicity labels of the train texts.
    :param dummy_y: the train dummy labels (used to compute the dummy loss).
    :param val_texts: the tokenized val texts.
    :param y: the toxicity labels of the val texts.
    :param val_dummy_y: the val dummy labels (used to compute the dummy val loss).
    :return: the history of the model's training
    """

    # Create vocab and lookup tables
    self.create_vocab(tokenized_texts)
    
    # turn the tokenized texts and token labels to padded sequences of indices
    X = self.to_sequences(tokenized_texts)
    dummy_y = pad_sequences(maxlen=self.max_seq_len, sequences=dummy_y, padding='post', value=0.0, dtype='float32')
    # build the model and compile it
    self.build(self.vocab_size)
    # start training
    vx = self.to_sequences(val_texts)
    val_dummy_y = pad_sequences(maxlen=self.max_seq_len, sequences=val_dummy_y, padding='post', value=0.0, dtype='float32')
    history = self.model.fit(X, [y, dummy_y], batch_size=self.batch_size,
                             epochs=self.epochs, validation_data=(vx,[val_y, val_dummy_y])
                              , verbose=1, callbacks=[self.earlystop])
    return history
  
  def predict(self, tokenized_texts):
        predictions, atts = self.model.predict(self.to_sequences(tokenized_texts))
        return predictions, atts
  
  def finetune_att_threshold(self, val_tokenized_texts, val_token_offsets, val_position):
    """
    This method finetunes the attention threshold of the model on the validation data based on the F1 score.
    :param val_tokenized_texts: the validation tokenized texts.
    :param val_token_offsets: a list that contains lists, where each list contains the character offsets of each token (of a sequence).
    :param val_position:  the ground truth (spans) of the val set.
    :return: the best threshold according to the best F1 score (calculated on the val data).
    """
    f1s = []
    for th in range(0,100):
      th = th/100
      pred_offsets = self.get_toxic_offsets(val_tokenized_texts, threshold=th)
      pred_char_offsets = self.get_toxic_char_offsets(val_token_offsets, pred_offsets)
      f1 = np.mean([semeval2021.f1(p,g) for p,g in list(zip(pred_char_offsets, val_position))])
      f1s.append(f1)
    best_th = np.argmax(f1s)/100 #best threshold results to highest f1 score 
    print('Optimal threshold is: ',best_th, ' with F1 score = ',max(f1s))
    return best_th

  def get_toxic_offsets(self, tokenized_texts, threshold=0.5):
    """
    This method labels the tokens of each sequence as toxic or not.
    :param tokenized_texts: the tokenized texts from which the toxic spans will be extracted.
    :param threshold: the attention threshold. If a token's attention is higher than this threshold, then the token is labeled as toxic.
    :return: the toxicity labels of each token in the text.
    """
    text_predictions, atts = self.predict(tokenized_texts)
    output = []
    for tokens, scores in list(zip(tokenized_texts, atts)): #use the att weights to classify each token as toxic or not 
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
        
        

class BERT_ARE():
  """
  Bert toxicity classifier.
  To detect toxic spans, we used the attention scores (from the ['CLS'] to the other tokens) from the heads of BERT’s last layer averaged
  over the heads, respectively. We obtain a sequence of binary decisions (toxic, non-toxic) for the tokens of the post (inherited by their
  character offsets) by using a probability threshold (tuned on development data) applied to the attention scores. 
  """

  def __init__(self,
               trainable_layers=3,
               max_seq_length=128,
               show_summary=True,
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
                          Recall(name='recall')
                        ]     
                 ):
        self.session = session
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False,  max_length=max_seq_length,pad_to_max_length=True)
        self.lr = lr
        self.batch_size = batch_size
        self.trainable_layers = trainable_layers
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


  #prepare inputs for bert 
  def to_bert_input(self, texts):
    """ 
    This method returns the inputs that Bert needs: input_ids, input_masks, input_segments 
    :param texts: the texts that will be converted into bert inputs 
    :return: a tuple containing the input_ids, input_masks, input_segments 
    """
    input_ids, input_masks, input_segments = [],[],[]
    for text in tqdm(texts):
      inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                  return_attention_mask=True, return_token_type_ids=True)
      input_ids.append(inputs['input_ids'])
      input_masks.append(inputs['attention_mask'])
      input_segments.append(inputs['token_type_ids'])
    return (np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32'))



  def build(self):
    """
    This method builds the model (defining the model's architecture).
    """

    in_id = Input(shape=(self.max_seq_length,), name='input_ids', dtype='int32')
    in_mask = Input(shape=(self.max_seq_length,), name='input_masks', dtype='int32')
    in_segment = Input(shape=(self.max_seq_length,), name='segment_ids', dtype='int32')
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = self.BERT(bert_inputs).last_hidden_state
    bert_output = bert_output[:,0,:] #take only the ['CLS'] token
    pred = Dense(1, activation = self.dense_activation, name = 'classification')(bert_output)
    self.model = tensorflow.keras.models.Model(inputs=bert_inputs, outputs=pred)
    self.model.compile(loss=self.loss,
                      optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=self.METRICS)
    if self.show_summary:
      self.model.summary()
  
  
  def fit(self, train_X, train_y, val_X, val_y):
    """
    This method fits the model.
    :param train_X: the train texts on which the model will be trained.
    :param train_y: the toxicity labels of the train texts.
    :param val_X: the val texts on which the model will be evaluated.
    :param val_y: the toxicity labels of the val texts.
    :return: the history of the model's training
    """

    train_input = self.to_bert_input(train_X)
    val_input = self.to_bert_input(val_X)
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
        return predictions
  
  def compute_atts(self, texts, layer = 12):
    """
    This method computes the average (of all heads) attention of a given layer for the '[CLS]' token.
    :param texts: the texts from which the attention scores will be extracted.
    :layer: the layer of Bert from which the attention scores will be extracted.
    """

    input = self.to_bert_input(texts)
    bert_atts = tensorflow.math.reduce_mean(self.BERT(input).attentions[layer-1][:,:], axis=1)[:,0,:]
    return bert_atts
  
  def get_attentions(self, test_texts, layer = 12, batch_size = 100):
    """
    This method extracts the attentions (average of all heads from a given layer) for the '[CLS]' token.
    :param test_texts: the texts from which the attention scores will be extracted.
    :layer: the layer of Bert from which the attention scores will be extracted.
    :batch_size: we extract the attention scores batch-wise in order no to get OOM error.
    """
    atts = []
    counter = 0 
    if len(test_texts) < batch_size:
      texts = test_texts
      atts.append(self.compute_atts(texts, layer = 12))
    else:
      for i in range(0, len(test_texts)//batch_size):
        texts = test_texts[i*batch_size: (i+1)*batch_size]
        atts.append(self.compute_atts(texts, layer = 12))
        counter = i
      counter +=1
      if  len(test_texts)%batch_size != 0: #do the remaininig batch 
        texts = test_texts[counter*batch_size:]
        atts.append(self.compute_atts(texts, layer = 12))
    atts = [item for sublist in atts for item in sublist] #flatten
    return atts

  def finetune_att_threshold(self, val_texts, val_position):
    """
    This method finetunes the attention threshold of the model on the validation data based on the F1 score.
    :param val_texts: the validation texts.
    :param val_position:  the ground truth (spans) of the val set.
    :return: the best threshold according to the best F1 score (calculated on the val data).
    """

    f1s = [] #save f1s for all possible thresholds
    for th in range(0,100):
      th = th/100
      pred_offsets = self.get_toxic_offsets(val_texts, threshold=th)
      pred_char_offsets = self.get_toxic_char_offsets(val_texts, pred_offsets)
      f1 = np.mean([semeval2021.f1(p,g) for p,g in list(zip(pred_char_offsets, val_position))])
      f1s.append(f1)
    best_th = np.argmax(f1s)/100 #best threshold results to highest f1 score 
    print("Optimal threshold is: ",best_th, " with F1 score = ",max(f1s))
    return best_th
  
  #label the tokens of each sequence as toxic or not 
  def get_toxic_offsets(self, texts, threshold=0.5):
    """
    This method labels the tokens of each sequence as toxic or not.
    :param texts: the texts from which the toxic spans will be extracted.
    :param threshold: the attention threshold. If a token's attention is higher than this threshold, then the token is labeled as toxic.
    :return: the toxicity labels of each token in the text.
    """

    att_weights = self.get_attentions(texts, layer = 12, batch_size = 100)
    output = []
    tokenized_texts = [['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]'] for text in texts]
    for i, scores in enumerate(att_weights): #use the att weights to classify each token as toxic or not 
      start = 0  
      end = min(len(tokenized_texts[i]),self.max_seq_length) #ignore pad tokens
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
      for j,token_label in enumerate(toxic_offsets[i]):
        if j == 0 or (j == len(tokens)-1 and len(tokens) <= self.max_seq_length): #ignore ['CLS'] and ['SEP'] tokens
            continue
        if token_label == 1:  #if token (or subtoken) was found toxic by the classifier
          index_of_word = tokenized_text.token_to_word(j) #get the index of the word in the sequence 
          (start, end) = tokenized_text.word_to_chars(index_of_word) #get the char offsets of the first and last chars of this token (subtoken)
          instance_toxic_char_offsets.extend([ch for ch in range(start,end)]) #add the char offsets of this token (subtoken) to the list
      toxic_char_offsets.append(set(instance_toxic_char_offsets)) #set removes the duplicates char offsets (if a subtoken was found toxic, we add the char offsets of the whole word)
    return toxic_char_offsets

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)