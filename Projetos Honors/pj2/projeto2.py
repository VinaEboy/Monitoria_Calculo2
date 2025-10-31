import numpy as np
import matplotlib.pyplot as plt

def fd_error(f,df,x0,h0,hn,n):
    x, y = [], []
    
    #retorna um vetor hi com espaçamento linear
    #são (n+1) pontos, então o último paramêtro é n+1
    hi = np.logspace(np.log10(h0), np.log10(hn), n+1)

    #para cada hi, calcula o valor da derivada aproximado por diferença finita avançada 
    #e calcula o erro em relação a derivada real
    for i in range (0,n+1):
        df_aprox = ( f(x0 + hi[i]) - f(x0) )/hi[i]
        erro = abs(df(x0) - df_aprox)
        x.append(hi[i])
        y.append(erro)

    x = np.array(x)
    y = np.array(y)

    plt.plot(x, y)
    plt.xscale('log')  
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.show()


def ode_solver(f,x0,y0,xn,n,plot):
    x, y = [], []

    #esse passo é linear
    passo = (xn - x0)/n
    xi = x0
    yi = y0

    #vai fazendo o método de Euler e armazenando nos vetores xi e yi
    for i in range(0, n+1):
        x.append(xi)
        y.append(yi)
        yi += passo*f(xi,yi)
        xi += passo

    x = np.array(x)
    y = np.array(y)

    if plot:
        plt.plot(x, y)
        plt.show()

    return x, y