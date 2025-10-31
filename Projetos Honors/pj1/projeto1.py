def trapezio(f,a,b,n):
    soma = f(a) + f(b) # X0 e Xn é somado apenas uma vez
    h = (b-a)/n
    for i in range(n-1):
        soma += 2*f(a + (i+1)*h)
    soma = soma*(h/2)
    return soma 

def simpson(f,a,b,n):
    soma = f(a) + f(b)  # X0 e Xn é somado apenas uma vez
    h = (b-a)/n
    for i in range(int(n/2 -1)): # n tem que ser par, senão é arredonado para baixo
        soma += 2*f(a+2*(i+1)*h) #começa no X2 e vai indo de 2 em 2
    for i in range(int(n/2)):
        soma += 4*f(a+(2*i+1)*h) #começa no X1 e vai indo de 2 em 2
    soma = soma*(h/3) 
    return soma
