import math
from projeto1 import trapezio
from projeto1 import simpson

n = int(input())

def f(x):
    return pow(x,n)*pow(math.e,-x)

print(trapezio(f,0,n*10,100*n))
print(simpson(f,0,n*10,100*n))
print(trapezio(lambda x: x**5,0,1,90))
print(simpson(lambda x: x**5,0,1,90))
print(simpson(lambda x: pow(1+x,x),0,1,10))