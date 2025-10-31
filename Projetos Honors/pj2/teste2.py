import projeto2 as p2 
import numpy as np 
import matplotlib.pyplot as plt
import math as m


x1,y1 = p2.ode_solver(lambda x,y: np.exp(-x**2),-3,-0.8862073482595,3,500,True)

x2,y2 = p2.ode_solver(lambda x,y: 10*np.sqrt(y)*np.sin(x)+x,0,0,100,500,True)

x3,y3 = p2.ode_solver(lambda x,y: np.cos(x**2),0.5,2,5,4,True)
print(np.array([x3,y3]).transpose())

#valores do eixo a e b de uma elipse
a = 1
b = 2
x4, x5 = p2.ode_solver(lambda x,y: m.pow(m.pow(a,2)*m.pow(m.sin(x),2)+m.pow(b,2)*m.pow(m.cos(x),2), 0.5), 0, 0, m.pi*2, 1000, True)
#não existe fórmula para o arco da elipse, não existe integral, 
#então só é possível chegar no valor númerico 
#aproximação bem ruinzinha é (a+b)*pi para toda uma volta