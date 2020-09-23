import numpy as np
from sklearn.naive_bayes import MultinomialNB


MODEL = MultinomialNB()

class NameBanker:


    def __init__(self):
        self.model = MODEL
    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        self.data = [X, y]
        self.model.fit(X, y)


    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        return self.model.predict_proba(np.array(x).reshape(1, -1))

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
        repaid = np.where(probs[0] == max(probs[0]))

        return U[action, repaid]


    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.

    def get_best_action(self, x):
        util = [self.expected_utility(x, a) for a in [0,1]]
        return util.index(max(util))
