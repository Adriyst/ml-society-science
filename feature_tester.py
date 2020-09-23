import pandas
import numpy as np

PATH = "../../data/credit/D_valid.csv"

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'

df = pandas.read_csv(PATH, sep=' ',
                     names=features+[target])


import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations

train, test = train_test_split(X)

perms = combinations(encoded_features, 2)
permlist = [x for x in perms]
models = [LogisticRegression() for _ in range(len(permlist))]

scores = []
for m,f in zip(models,permlist):
    m.fit(train[[f[0], f[1]]], train[target])

train_scores = np.array([accuracy_score(test[target], m.predict(test[[f[0], f[1]]])) for m,f in zip(models, permlist)])

twentyfifth = np.percentile(train_scores, 99).mean()
best_idx = np.where(train_scores >= twentyfifth)


chosen_perms, chosen_scores = np.array(permlist)[best_idx], train_scores[best_idx]
chosen_perms = [",".join(x) for x in chosen_perms]


plt.barh(range(len(chosen_perms)), chosen_scores)
plt.yticks(range(len(chosen_perms)), chosen_perms)
plt.show()

#nb = MultinomialNB()
#nb.fit(train[["checking account balance"]], train[target])
#res = nb.predict(test[["checking account balance"]])
#print(sum((res == test[target]).astype("int"))/len(res))
