#-*-coding:utf-8-*-
'''
打印时间日志,如果有日志文件，那么输入到日志文件中，如果没有，直接打印出来
'''
import datetime

def printTimeLog(X, f = None):
    time_stamp = datetime.datetime.now().strftime("")
    if not f:
        print(time_stamp + " " + X )
    else:
        f.write(time_stamp + " " + X)


