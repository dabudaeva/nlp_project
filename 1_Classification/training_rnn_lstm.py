import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 90446
LEARNING_RATE = 0.005
BATCH_SIZE = 50
NUM_EPOCHS = 2
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_CLASSES = 1

df = pd.read_csv('data/clean.csv')
features = np.load('data/features.npy')
corpus = pickle.load(open('data/corpus.pkl', 'rb'))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
x_train = features[:35000]
y_train = df['sentiment'].iloc[:35000]
x_valid = features[35000:]
y_valid = df['sentiment'][35000:]

train_ldr = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.values)), shuffle=True, batch_size=50)
test_ldr = DataLoader(TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid.values)), shuffle=True, batch_size=50)


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

torch.manual_seed(RANDOM_SEED)
model = RNN(input_dim=VOCABULARY_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES # could use 1 for binary classification
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()
history = None
def model_train(model, epochs, history=None):
    history = history or {'train_accs': [], 'train_losses': [], 'test_accs': [], 'test_losses': []}
    start_epoch = len(history['train_accs'])
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        print(f'{"-" * 13} Epoch {epoch} {"-" * 13}')
        model.train()
        batch_accs = []
        batch_losses = []
        for x_train_batch, y_train_batch in train_ldr:
            print('-', end='')
            y_pred = model(x_train_batch.reshape([x_train_batch.shape[1], x_train_batch.shape[0]]))
            loss = criterion(torch.round(y_pred).squeeze(), y_train_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batch_accs.append((torch.round(y_pred).squeeze() == y_train_batch).numpy().mean())
        print('end!')
        history['train_losses'].append(np.mean(batch_losses))
        history['train_accs'].append(np.mean(batch_accs))

        model.eval()
        batch_accs = []
        batch_losses = []
        for x_test_batch, y_test_batch in test_ldr:
            print('-', end='')
            y_pred = model(x_test_batch.reshape([x_test_batch.shape[1], x_test_batch.shape[0]]))
            #y_pred = model(x_test_batch)
            loss = criterion(torch.round(y_pred).squeeze(), y_test_batch.float())
            batch_losses.append(loss.item())
            batch_accs.append((torch.round(y_pred).squeeze() == y_test_batch).numpy().mean())
        history['test_losses'].append(np.mean(batch_losses))
        history['test_accs'].append(np.mean(batch_accs))
        print('end!')
        print(f"train_loss={history['train_losses'][-1]:.3f}, "
              f"valid_loss={history['test_losses'][-1]:.3f}, "
              f"train_accs={history['train_accs'][-1]:.3f}, "
              f"valid_accs={history['test_accs'][-1]:.3f}")
    return history

history = model_train(model, 2, history)


torch.save(model.state_dict(), 'models/rnn_weights.pt')

# loaded_model = RNN(input_dim=VOCABULARY_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=NUM_CLASSES)
# loaded_model.load_state_dict(torch.load('models/rnn_weights.pt'))

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('data/clean.csv')
features = np.load('data/features.npy')
corpus = pickle.load(open('data/corpus.pkl', 'rb'))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
x_train = features[:35000]
y_train = df['sentiment'].iloc[:35000]
x_valid = features[35000:]
y_valid = df['sentiment'][35000:]

train_ldr = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.values)), shuffle=True, batch_size=50)
test_ldr = DataLoader(TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid.values)), shuffle=True, batch_size=50)

batch_size = 50

class RNNNet(nn.Module):
    '''
    vocab_size: int, размер словаря (аргумент embedding-слоя)
    emb_size:   int, размер вектора для описания каждого элемента последовательности
    hidden_dim: int, размер вектора скрытого состояния
    batch_size: int, размер batch'а
    '''

    def __init__(self, vocab_size, emb_size, hidden_dim, seq_len, batch_size) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.rnn_cell = nn.RNN(self.emb_size, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim * self.seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        rnn_out, h_out = self.rnn_cell(x, self.init_hidden())
        print(f'rnn_out {rnn_out.shape}')
        print(f'hidden out {h_out.shape}')
        x = rnn_out.contiguous().view(rnn_out.size(0), -1)
        self.x_after_view = x.size(1)
        print(f'x size after view {x.shape}')
        linear_out = self.linear(x)
        output = torch.sigmoid(linear_out)
        return output

    def init_hidden(self):
        self.h_0 = torch.zeros((1, self.batch_size, self.hidden_dim))
        return self.h_0



vocab_size = len(corpus) + 1
output_size = 1
embedding_dim = 200
hidden_dim = 32

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = len(corpus) + 1
output_size = 1
embedding_dim = 200
hidden_dim = 50
n_layers = 4

model = sentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model = model.to(device)

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

clip = 5
epochs = 2
valid_loss_min = np.Inf

epoch_tr_loss, epoch_vl_loss = [],[]
epoch_tr_acc, epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # инициализируем хидден стейты и селл стейты
    h = model.init_hidden(50)
    for inputs, labels in train_ldr:
        print('-', end='')
        inputs, labels = inputs.to(device), labels.to(device)
        # хидден стейты необходимо инициализировать каждый раз заново
        # иначе backprop не справится (а нам и надо)
        h = tuple([each.data for each in h])

        optimizer.zero_grad()
        output, h = model(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())

        accuracy = acc(output, labels)
        train_acc += accuracy
        # можно использовать клиппинг
        # nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in test_ldr:
        val_h = tuple([each.data for each in val_h])

        inputs, labels = inputs.to(device), labels.to(device)

        output, val_h = model(inputs, val_h)
        val_loss = criterion(output.squeeze(), labels.float())

        val_losses.append(val_loss.item())

        accuracy = acc(output, labels)
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_ldr.dataset)
    epoch_val_acc = val_acc / len(test_ldr.dataset)

    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)

    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss:.4f} val_loss : {epoch_val_loss:.4f}')
    print(f'train_accuracy : {epoch_train_acc * 100:.2f} val_accuracy : {epoch_val_acc * 100:.2f}')

    # сохранение модели

torch.save(model.state_dict(), 'models/lstm_weights.pt')
