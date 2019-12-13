#-*-coding:utf-8-*-
'''
辛普森算法：用已知的函数f来进行数值积分。公式如下
int(f) = delta_x/2 * (b-a)/3*(f1 + 4f2 + 2f_3 + ... + fn)
'''
def SimpsonRule(boundary, steps):
# "Simpson Rule"
# int(f) = delta_x/2 * (b-a)/3*(f1 + 4f2 + 2f_3 + ... + fn)
	h = (boundary[1] - boundary[0]) / steps
	a = boundary[0]
	b = boundary[1]
	x_i = makePoints(a,b,h)
	y = 0.0
	y += (h/3.0)*f(a)
	cnt = 2
	for i in x_i:
		y += (h/3)*(4-2*(cnt%2))*f(i)	
		cnt += 1
	y += (h/3.0)*f(b)
	return y

def makePoints(a,b,h):
	x = a + h
	while x < (b-h):
		yield x
		x = x + h

def f(x): #这里属于你自己想定义的函数
	y = (x-0)*(x-0)
	return y

def main():
	steps = 10.0		#define number of steps or resolution
	boundary = [0, 1]	#定义integration的上下限
	y = SimpsonRule(boundary, steps)
	print('y = {0}'.format(y))

if __name__ == '__main__':
        main()