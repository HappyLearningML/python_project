#-*- coding:utf-8-*-
'''
牛顿方法，梯度下降法一种，这里仅提供了简单的数值例子
'''
def newton(function,function1,startingInt): #function is the f(x) and function1 is the f'(x)
  x_n=startingInt
  while True:
      x_n1=x_n-function(x_n)/function1(x_n)
      if abs(x_n-x_n1)<0.00001:
          return x_n1
      x_n=x_n1

if __name__ == '__main__':
    def f(x):
        return (x**3)-2*x-5

    def f1(x):
        return 3*(x**2)-2

    print(newton(f,f1,3))