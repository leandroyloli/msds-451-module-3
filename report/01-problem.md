# Predicting the next-day direction of WEGE3.SA

MSDS 451, Programming Assignment 1. The asset is WEG ON, ticker WEGE3.SA.

# 1. Problem

The task is to predict the direction of the next session's close-to-close log return. The series is WEGE3.SA, the common share of WEG S.A. listed on B3. It is not the WTI series used in the course example.

On day t the label is 1 when log(C(t) / C(t-1)) is positive, and 0 when the close is flat or down. A forecast for day t may use only prices and volume known at the close of day t-1. The open, high, low, close, and volume of day t itself are not predictors.

The course example fits an XGBoost classifier and then scores that fit on the same rows used to train it. This report keeps a later block of sessions out of feature selection and out of the hyperparameter search. The finished classifier is scored on that block once.
