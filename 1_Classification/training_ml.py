# Loading packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
# from catboost import CatBoostClassifier
import pickle


# Loading data
df = pd.read_csv('data/clean.csv')

cvec = CountVectorizer(ngram_range=(1, 1)).fit(df['review'])
tfid = TfidfVectorizer(ngram_range=(1, 1)).fit(df['review'])
cvec_df = cvec.transform(df['review'])
tfid_df = tfid.transform(df['review'])

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
x_train_tfid = tfid_df[:35000]
x_train_cvec = cvec_df[:35000]
y_train = df['sentiment'].iloc[:35000]
x_valid_tfid = tfid_df[35000:]
x_valid_cvec = cvec_df[35000:]
y_valid = df['sentiment'][35000:]


# LogisticRegression
lr_cvec = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=100).fit(x_train_cvec, y_train)
accuracy_score(y_valid, lr_tfid.predict(x_valid_cvec)) # 0.8768

lr_tfid = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=100).fit(x_train_tfid, y_train)
accuracy_score(y_valid, lr_tfid.predict(x_valid_tfid)) # 0.8944

lr_cvec = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=100).fit(cvec_df, df['sentiment'])
lr_tfid = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=100).fit(tfid_df, df['sentiment'])
pickle.dump(lr_cvec, open('models/logistic_cvec.sav', 'wb'))
pickle.dump(lr_tfid, open('models/logistic_tfid.sav', 'wb'))


# Random forest
rf_cvec = RandomForestClassifier(n_estimators=1000, max_depth=5).fit(x_train_cvec, y_train)
accuracy_score(y_valid, rf_cvec.predict(x_valid_cvec)) # 0.8499

rf_tfid = RandomForestClassifier(n_estimators=1000, max_depth=5).fit(x_train_tfid, y_train)
accuracy_score(y_valid, rf_tfid.predict(x_valid_cvec)) # 0.8377


rf_cvec = RandomForestClassifier(n_estimators=1000, max_depth=5).fit(cvec_df, df['sentiment'])
rf_tfid = RandomForestClassifier(n_estimators=1000, max_depth=5).fit(tfid_df, df['sentiment'])
pickle.dump(rf_cvec, open('models/random_forest_cvec.sav', 'wb'))
pickle.dump(rf_tfid, open('models/random_forest_tfid.sav', 'wb'))

# # Catboost
# catboost_params = {
#     'loss_function': 'CrossEntropy',
#     'iterations': 1000,
#     'learning_rate': 0.01,
#     'eval_metric': 'Accuracy',
# }
# cb = CatBoostClassifier(**catboost_params).fit(x_train, y_train)
# y_pred = cb.predict(x_valid)
# confusion_matrix(y_valid, y_pred)
# accuracy_score(y_valid, y_pred) # 0.8328
#
# cb = CatBoostClassifier(**catboost_params).fit(tfid_df, df['sentiment'])
# cb.save_model('models/catboost')

