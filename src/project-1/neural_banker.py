import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.optim import SGD
import torch
import random

MODEL = nn.Sequential(
    nn.Linear(50, 10),
    nn.ReLU(),
    nn.Linear(10, 2),
    nn.ReLU()
)

LEARN_RATE = 0.01

LOSS_FN = nn.CrossEntropyLoss()
optimizer = SGD(MODEL.parameters(), lr=LEARN_RATE)

BATCH_SIZE = 8

EPOCHS = 5

class NeuralBanker:


    def __init__(self):
        self.model = MODEL
    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        self.data = [X, y]
        MODEL.train()
        random_idx = np.random.permutation(X.index)
        X = X.reindex(random_idx)
        y = y.reindex(random_idx)
        batched_X = np.array_split(X, len(X)/BATCH_SIZE)
        batched_y = np.array_split(y, len(y)/BATCH_SIZE)
        for _ in range(EPOCHS):
            running_loss = 0.0
            for x,lab in zip(batched_X, batched_y):
                x = torch.FloatTensor(np.array(x))
                # substract to normalize labels from [1,2] to [0,1]
                lab = torch.LongTensor(np.array(lab)) - 1
                MODEL.zero_grad()
                preds = MODEL(x)
                loss = LOSS_FN(preds, lab)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            #print(f"LOSS: {running_loss / len(X)/BATCH_SIZE}")


    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        #return self.model.predict_proba(np.array(x).reshape(1,-1))
        with torch.no_grad():
            preds = MODEL(torch.FloatTensor(np.array(x)))
        print(preds)
        return preds

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        utility = x["amount"] * ((1 + self.rate) ** x['duration'])
        U = np.matrix(f"0 0; {utility} {x['amount'] * -1}")

        probs = self.predict_proba(x)
        repaid = np.where(probs == max(probs))
        if len(repaid[0]) > 1:
            repaid = 0 if random.random() > .5 else 1

        return U[action, repaid]


    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.

    def get_best_action(self, x):
        util = [self.expected_utility(x, a) for a in [0,1]]
        return util.index(max(util))
