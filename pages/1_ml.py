import streamlit as st

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

df = pd.read_csv('data/clean.csv')
cvec = CountVectorizer(ngram_range=(1, 1)).fit(df['review'])
tfid = TfidfVectorizer(ngram_range=(1, 1)).fit(df['review'])

lr_cvec = pickle.load(open('models/logistic_cvec.sav', 'rb'))
lr_tfid = pickle.load(open('models/logistic_tfid.sav', 'rb'))
rf_cvec = pickle.load(open('models/random_forest_cvec.sav', 'rb'))
rf_tfid = pickle.load(open('models/random_forest_tfid.sav', 'rb'))


# cb = CatBoostClassifier()
# cb.load_model('models/catboost')
# cb.predict_proba(tfid.transform(['This movie is awesome']))

st.write('''# Классификация с помощью классических ML-алгоритмов''')
from PIL import Image
image = Image.open('images/Figure_1.png')
st.image(image, caption='', use_column_width='auto')

st.write('''
- Логистическая регрессия
    - на BagOfWords результат показал **87.68 %** точности
    - на TF-IDF результат показал **89.44 %** точности
- Метод Случайных лесов
    - на BagOfWords результат показал **84.99 %** точности
    - на TF-IDF результат показал **83.77 %** точности
''')


def predict(model, text):
    if model == lr_cvec or model == rf_cvec:
        pred = model.predict(cvec.transform([text]))
        proba = model.predict_proba(cvec.transform([text]))
    if model == lr_tfid or model == rf_tfid:
        pred = model.predict(tfid.transform([text]))
        proba = model.predict_proba(tfid.transform([text]))

    if pred == 0:
        return f"Негативный ({round(proba[0][0]*100, 2)} %)"
    if pred == 1:
        return f"Позитивный ({round(proba[0][1]*100, 2)} %)"

st.markdown('#### Проверим на вашем отзыве?')

text = st.text_area('Напишите свой отзыв сюда (на английском):', value="This movie is awesome", height=200)

st.write(
pd.DataFrame({'BagOfWords': [predict(lr_cvec, text), predict(rf_cvec, text)],
              'TF-IDF': [predict(lr_tfid, text), predict(rf_tfid, text)]},
             index=['Логистическая регрессия', 'Метод Случайных лесов']))