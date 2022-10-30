# Loading packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from collections import Counter
import pickle

# Loading data
df = pd.read_csv('data/IMDB Dataset.csv')

# Preprocessing
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])  # 1 - positive, 0 - negative

sw = stopwords.words('english')
sw += ['br', 'film', 'movie', 'story', 'show', 'one', 'character', 'wa', 'ha', 'even', 'see']

reg_tokenizer = RegexpTokenizer('\w+')
tokenized = reg_tokenizer.tokenize_sents(df['review'])


def my_preprocessing(tokenized_data):
    data = []
    lemmatizer = WordNetLemmatizer()
    for i in tokenized:
        clean = list(map(lambda x: x.lower(), i))  # переводим в нижний регистр
        clean = [re.sub(r'\d', '', text) for text in clean]  # удаляем цифры
        clean = [lemmatizer.lemmatize(word) for word in clean]  # лемматизируем
        clean = [word for word in clean if not word in sw]  # удаляем стоп-слова
        if '' in clean:
            clean.remove('')
        data.append(clean)
    return data


clean = my_preprocessing(tokenized)

df['review'] = pd.Series([' '.join(clean[i]) for i in range(len(clean))])

# Visualization

pos = df[df['sentiment'] == 1]
neg = df[df['sentiment'] == 0]

pos = WordCloud().generate(' '.join(pos['review']))
neg = WordCloud().generate(' '.join(neg['review']))

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].axis('off')
ax[1].axis('off')
ax[0].imshow(pos)
ax[0].set_title('Positive')
ax[1].imshow(neg)
ax[1].set_title('Negative')
plt.show()

df.to_csv('data/clean.csv', index=False)


# Prepare data for RNN
corpus = Counter([word for text in df['review'] for word in text.split()])
corpus = corpus.most_common()
corpus = {w: i+1 for i, (w, c) in enumerate(corpus)}

features = []
for text in df['review']:
    r = [corpus[word] for word in text.split()]
    features.append(r)


def padding(features: list, seq_len: int) -> np.array:
    feature = np.zeros((len(features), seq_len), dtype=int)
    for i, review in enumerate(features):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        feature[i, :] = np.array(new)
    return feature

features = padding(features, 200)

pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
np.save('data/features', features)
