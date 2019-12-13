#-*-coding:utf-8-*-
'''
Trapezoidal算法：用已知的函数f来进行数值积分或者求积
公式如下：int(f) = dx/2 * (f1 + 2f2 + ... + fn)
和辛普森法则类似。只不过公式不一样
'''
def TrapezoidalRule(boundary, steps):
# "extended trapezoidal rule"
# int(f) = dx/2 * (f1 + 2f2 + ... + fn)
	h = (boundary[1] - boundary[0]) / steps
	a = boundary[0]
	b = boundary[1]
	x_i = makePoints(a,b,h)
	y = 0.0	
	y += (h/2.0)*f(a)
	for i in x_i:
		#print(i)	
		y += h*f(i)
	y += (h/2.0)*f(b)	
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
	y = TrapezoidalRule(boundary, steps)
	print('y = {0}'.format(y))

if __name__ == '__main__':
        main()