#-*-coding:utf-8-*-
'''
scoring function: 主要是计算预测值与真实值之间的距离的方法合集
'''

import numpy as np

#Mean Absolute Error
def mae(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    difference = abs(predict - actual)
    score = difference.mean()

    return score

#Mean Squared Error
def mse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    difference = predict - actual
    square_diff = np.square(difference)

    score = square_diff.mean()
    return score

#Root Mean Squared Error
def rmse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    difference = predict - actual
    square_diff = np.square(difference)
    mean_square_diff = square_diff.mean()
    score = np.sqrt(mean_square_diff)
    return score

#Root Mean Square Logarithmic Error
def rmsle(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    log_predict = np.log(predict+1)
    log_actual = np.log(actual+1)

    difference = log_predict - log_actual
    square_diff = np.square(difference)
    mean_square_diff = square_diff.mean()

    score = np.sqrt(mean_square_diff)

    return score

#Mean Bias Deviation
def mbd(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    difference = predict - actual
    numerator = np.sum(difference) / len(predict) 
    denumerator =  np.sum(actual) / len(predict)
    print(numerator)
    print(denumerator)

    score = float(numerator) / denumerator * 100

    return score