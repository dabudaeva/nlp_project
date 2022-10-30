import streamlit as st

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle

corpus = pickle.load(open('data/corpus.pkl', 'rb'))

class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        #self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.embedding = torch.nn.Embedding(input_dim+1, embedding_dim)
        self.rnn = torch.nn.RNN(embedding_dim,
                                hidden_dim,
                                nonlinearity='relu')
        # self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        #output, (hidden, cell) = self.rnn(embedded)
        output, hidden = self.rnn(embedded)
        hidden.squeeze_(0)
        output = torch.sigmoid(self.fc(hidden))
        return output

VOCABULARY_SIZE = 90446
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_CLASSES = 1

rnn = RNN(input_dim=VOCABULARY_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=NUM_CLASSES)
rnn.load_state_dict(torch.load('models/rnn_weights.pt'))

class sentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self,
                 # объем словаря, с которым мы работаем, размер входа для слоя Embedding
                 vocab_size,
                 # нейроны полносвязного слоя – у нас бинарная классификация - 1
                 output_size,
                 # размер выходного эмбеддинга каждый элемент последовательности
                 # будет описан вектором такой размерности
                 embedding_dim,
                 # размерность hidden state LSTM слоя
                 hidden_dim,
                 # число слоев в LSTM
                 n_layers,
                 drop_prob=0.5):
        super().__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=drop_prob,
                            batch_first=True)

        self.dropout = nn.Dropout()

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        # print(f'Embed shape: {embeds.shape}')
        lstm_out, hidden = self.lstm(embeds, hidden)
        # print(f'lstm_out {lstm_out.shape}')
        # print(f'hidden {hidden[0].shape}')
        # print(f'hidden {hidden[1].shape}')
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # print(f'lstm out after contiguous: {lstm_out.shape}')
        # Dropout and fully connected layer

        out = self.fc(lstm_out)
        out = self.dropout(out)

        # sigmoid function
        sig_out = self.sigmoid(out)

        # reshape to be batch size first
        # print(sig_out.shape)
        sig_out = sig_out.view(batch_size, -1)
        # print(sig_out.shape)
        # print(f'Sig out before indexing:{sig_out.shape}')
        sig_out = sig_out[:, -1]  # get last batch of labels
        # print(sig_out.shape)

        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Hidden state и Cell state инициализируем нулями '''
        # (число слоев; размер батча, размер hidden state)
        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

rnn = RNN(input_dim=VOCABULARY_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=NUM_CLASSES)
rnn.load_state_dict(torch.load('models/rnn_weights.pt'))

vocab_size = len(corpus) + 1
output_size = 1
embedding_dim = 200
hidden_dim = 50
n_layers = 4

lstm = sentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
lstm.load_state_dict(torch.load('models/lstm_weights.pt'))

from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()

def prepare(text):
    clean_text = text.lower()  # переводим в нижний регистр
    clean_text = re.sub(r'\d', '', clean_text)  # удаляем цифры
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = lemmatizer.lemmatize(clean_text)  # лемматизируем
    clean_text = clean_text.split()
    text_corpus = []
    for i in clean_text:
        if i in corpus.keys():
            text_corpus.append(corpus[i])
    text_corpus = np.array(text_corpus)
    seq_len = len(text_corpus)
    text_corpus = np.pad(text_corpus, (200 - len(text_corpus), 0))
    text_corpus = torch.Tensor(text_corpus).unsqueeze(0).to(torch.int64)
    return text_corpus, seq_len


st.write('''# Классификация с помощью RNN, LSTM моделей''')

text = st.text_area('Напишите свой отзыв сюда (на английском):', value="This movie is awesome", height=200)
text, seq_len = prepare(text)


rnn.eval()
pred = rnn(text)
cls = torch.round(pred.squeeze()[-seq_len:]).detach().numpy().mean()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lstm.eval()
val_h = lstm.init_hidden(batch_size=1)
val_h = tuple([each.data for each in val_h])
pred_lstm, hidden = lstm(text, val_h)
cls_lstm = pred_lstm.squeeze().item()#.detach().numpy()

if cls ==0:
    mark='Негативный'
if cls != 0:
    mark = 'Позитивный'

if round(cls_lstm) ==0:
    mark_lstm='Негативный'
if round(cls_lstm) != 0:
    mark_lstm = 'Позитивный'

st.write(pd.DataFrame({'Class': [mark, mark_lstm], 'Probability': [pred.squeeze()[-seq_len:].detach().numpy().mean(), cls_lstm]}, index=['RNN', 'LSTM']))