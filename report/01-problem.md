# 1. Problem

The task is to predict the direction of the next session's close-to-close log return. The series is WEGE3.SA. It is not the WTI series used in the course example.

WEG S.A. makes electric motors, generators, transformers, and industrial automation equipment. Its headquarters are in Jaraguá do Sul, in the state of Santa Catarina, Brazil. Its common share is listed on B3. B3 is the stock exchange of Brazil.

On day $t$ the label is 1 when the close-to-close log return is positive, and 0 when the close is flat or down:

$$y_t = \mathbf{1}\{\log(C_t / C_{t-1}) > 0\}.$$

A forecast for day $t$ may use only prices and volume known at the close of day $t-1$. The open, high, low, close, and volume of day $t$ itself are not predictors.

The course example fits an XGBoost classifier and then scores that fit on the same rows used to train it. This report keeps a later block of sessions out of feature selection and out of the hyperparameter search. The finished classifier is scored on that block once.
