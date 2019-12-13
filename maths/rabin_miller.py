#-*-coding:utf-8-*-
'''
Rabin-Miller Algorithm:一种常用的加密算法
Input
第一行：CAS,代表数据组数（不大于350），以下CAS行，每行一个数字，保证在64位长整形范围内，并且没有负数。你需要对于每个数字：第一，检验是否是质数，是质数就输出Prime 
第二，如果不是质数，输出它最大的质因子是哪个。 

Output
第一行CAS(CAS<=350，代表测试数据的组数) 
以下CAS行：每行一个数字，保证是在64位长整形范围内的正数。 
对于每组测试数据：输出Prime，代表它是质数，或者输出它最大的质因子，代表它是和数 
'''
# Primality Testing with the 

import random

def rabinMiller(num):
    s = num - 1
    t = 0

    while s % 2 == 0:
        s = s // 2
        t += 1

    for trials in range(5):
        a = random.randrange(2, num - 1)
        v = pow(a, s, num)
        if v != 1:
            i = 0
            while v != (num - 1):
                if i == t - 1:
                    return False
                else:
                    i = i + 1
                    v = (v ** 2) % num
    return True

def isPrime(num):
    if (num < 2):
        return False

    lowPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
                 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
                 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401,
                 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
                 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563,
                 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631,
                 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709,
                 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
                 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
                 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967,
                 971, 977, 983, 991, 997]

    if num in lowPrimes:
        return True

    for prime in lowPrimes:
        if (num % prime) == 0:
            return False

    return rabinMiller(num)

def generateLargePrime(keysize = 1024):
    while True:
        num = random.randrange(2 ** (keysize - 1), 2 ** (keysize))
        if isPrime(num):
            return num

if __name__ == '__main__':
    num = generateLargePrime()
    print(('Prime number:', num))
    print(('isPrime:', isPrime(num)))