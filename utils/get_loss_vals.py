#-*-coding:utf-8-*-
from collections import OrderedDict


def get_loss_vals(loss_store):
    return OrderedDict((key, val.eval()) for key,val in loss_store.items())