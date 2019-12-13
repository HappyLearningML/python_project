#-*-coding:utf-8-*-
'''
fmt的相关操作
'''
import re

def fmt_imsave(fmt, iteration):
    if re.match(r'^.*\{.*\}.*$', fmt):
        return fmt.format(iteration)
    elif '%' in fmt:
        return fmt % iteration
    else:
        raise ValueError("illegal format string '{}'".format(fmt))