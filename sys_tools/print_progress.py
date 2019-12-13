#-*-coding:utf-8-*-
'''
打印运行过程
'''
def print_progress(loss_vals):
    for key,val in loss_vals.items():
        print('{:>13s} {:g}'.format(key + ' loss:', val))
