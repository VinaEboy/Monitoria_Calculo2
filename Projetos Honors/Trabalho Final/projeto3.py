import numpy as np
import matplotlib.pyplot as plt
# Aluno Vinícius Gregorio Fucci, GRR20241272
# As funções testes estão no final
# Todos os achievements entregues : implementar tudo para N dimensões, 
# método de Newton, grad e hess por diferenças finitas,
# busca linear, salvaguardas e BFGS

# MACROS do programa referentes ao plot do gráfico
# os valores podem ser alterados dependendo do exemplo
ESPAÇAMENTO = 100
LEVELS = 70

# constantes
TAU = 1e-3
GAMMA = 0.5
AJUSTE_HESSIANA = 0.9
TOLERANCIA_ANGULO = -1e-3

def gd(f,x0,grad,eps = 1e-5 ,alpha = 0.1,itmax = 10000 ,fd = False,h = 1e-7 ,plot = False,search = False):
    if (fd):
        def grad(X):
            return fin_diff(f,X,1,h)
        
    pontos_x = [x0[0]] 
    pontos_y = [x0[1]]
    k = 0
    X = x0
    while (np.linalg.norm(grad(X)) > eps) and ( k < itmax):
        k = k + 1
        if (search):
            alpha = linesearch(f,X,grad,-grad(X))
        X = X - alpha*grad(X)
        pontos_x.append(X[0])
        pontos_y.append(X[1])

    if(plot):
        mostrar_grafico(np.array(pontos_x), np.array(pontos_y), f)

    return X, k


def newton(f,x0,grad, hess, eps = 1e-5 ,alpha = 0.1,itmax = 10000 ,fd = False,h = 1e-7 ,plot = False,search = False):
    if (fd):
        def grad(X):
            return fin_diff(f,X,1,h)
        def hess(X):
            return fin_diff(f,X,2,h)
        
    pontos_x = [x0[0]] 
    pontos_y = [x0[1]]
    k = 0
    X = x0
    g = grad(X)
    while (np.linalg.norm(g) > eps) and ( k < itmax):
        k = k + 1
        H = hess(X)

        try:
            d = np.linalg.solve(H,-g)
        except np.linalg.LinAlgError:
            H = AJUSTE_HESSIANA*H + (1-AJUSTE_HESSIANA)*np.eye(X.size)
            d = np.linalg.solve(H,-g)


        while (d @ g > TOLERANCIA_ANGULO*np.linalg.norm(g)*np.linalg.norm(d)):
            H = AJUSTE_HESSIANA*H + (1-AJUSTE_HESSIANA)*np.eye(X.size) #np.eye é a matriz identidade n por n
            d = np.linalg.solve(H,-g)   

        if (search):
            alpha = linesearch(f,X,grad,d)
        X = X + alpha*d
        pontos_x.append(X[0])
        pontos_y.append(X[1])
        g = grad(X)

    if(plot):
        mostrar_grafico(np.array(pontos_x), np.array(pontos_y), f)

    return X, k


def fin_diff(f,x,degree,h):
    n = x.size

    if (degree == 1):
        grad = np.zeros(n)
        for i in range(n):
            hei = np.zeros(n)
            hei[i] = h
            f_linha = (f(x + hei) - f(x - hei))/(2*h)
            grad[i] = f_linha
        
        return grad

    if (degree == 2):
        hess = np.zeros((n,n))
        for i in range(n):
            j = i
            #calcula apenas metade da matriz
            while (j < n):
                hej = np.zeros(n)
                hej[j] = h
                if (i == j):
                    f_linha = (f(x + hej) - 2*f(x) + f(x - hej) )/(h*h)
                else:
                    hei = np.zeros(n)
                    hei[i] = h
                    f_linha = ( f(x+hei+hej) - f(x+hei-hej) - f(x-hei+hej) + f(x-hei-hej))/(4*h*h)
                #assume que a hessiana é simétrica
                hess[i][j] = f_linha
                hess[j][i] = f_linha
                j = j + 1

        return hess


def linesearch(f,x,g,d):       
    alpha = 1
    while (f(x+alpha*d) > f(x) + alpha*TAU*(g(x) @ d)):
        alpha = alpha*GAMMA
    
    return alpha


def bfgs(f, x0, grad, hess, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=False, search=False):
    if fd:
        def grad(X):
            return fin_diff(f, X, 1, h)
        def hess(X):
            return fin_diff(f, X, 2, h)

    pontos_x = [x0[0]]
    pontos_y = [x0[1]]
    k = 0
    X = x0
    g = grad(X)
    H = np.eye(X.size)
    while (np.linalg.norm(g) > eps) and (k < itmax):
        k = k + 1
        d = -H @ g
        
        while (d @ g > TOLERANCIA_ANGULO * np.linalg.norm(g) * np.linalg.norm(d)):
            H = AJUSTE_HESSIANA*H + (1 - AJUSTE_HESSIANA)*np.eye(X.size)
            d = -H @ g
        
        if search:
            alpha = linesearch(f, X, grad, d)
        
        s = X
        y = g
        X = X + alpha * d
        g = grad(X)
        s = X - s
        y = g - y

        #no exemplo 3 não estava convergindo devido a imprecisão na divisão
        denom = np.dot(s, y)
        if abs(denom) < eps:
            denom = eps
        
        H = H + (np.dot(s,y) + (y@H@y))*(np.outer(s,s))/(denom**2) - (H@(np.outer(y,s))+(np.outer(s,y)@H))/(denom)
        
        pontos_x.append(X[0])
        pontos_y.append(X[1])
    
    if plot:
        mostrar_grafico(np.array(pontos_x), np.array(pontos_y), f)
    
    return X, k
        

def mostrar_grafico(pontos_x, pontos_y, f):

    #mostra 10% para esquerda, direita, cima e baixo, para além do caminho desenhado
    AJUSTE_X = (np.max(pontos_x) - np.min(pontos_x))*0.1
    AJUSTE_Y = (np.max(pontos_y) - np.min(pontos_y))*0.1

    limite_x_inferior = np.min(pontos_x) - AJUSTE_X
    limite_x_superior = np.max(pontos_x) + AJUSTE_X
    limite_y_inferior = np.min(pontos_y) - AJUSTE_Y
    limite_y_superior = np.max(pontos_y) + AJUSTE_Y

    #verifica se alguns dos limites do gráfico seria Inf ou NaN
    if any(not np.isfinite(i) for i in [limite_x_inferior, limite_y_inferior, limite_x_superior, limite_y_superior]):
         return

    intervalo_x = np.linspace(limite_x_inferior, limite_x_superior,ESPAÇAMENTO)
    intervalo_y = np.linspace(limite_y_inferior,limite_y_superior,ESPAÇAMENTO)
     
    coord_X, coord_Y = np.meshgrid(intervalo_x, intervalo_y)

    coord_Z = []
    for i in range(ESPAÇAMENTO):
        for j in range(ESPAÇAMENTO):
            #combina a coord_X e coord_Y para calcular um ponto P
            P = np.array([ coord_X[i,j], coord_Y[i, j] ])
            coord_Z.append(f(P))
        
    coord_Z = np.array(coord_Z)
    coord_Z = coord_Z.reshape(ESPAÇAMENTO,ESPAÇAMENTO)

    plt.plot(pontos_x, pontos_y, color='black', linestyle='-', linewidth=2, zorder=4)
    plt.xlim(limite_x_inferior,limite_x_superior)
    plt.ylim(limite_y_inferior,limite_y_superior)
    plt.contour(coord_X, coord_Y, coord_Z, LEVELS)
    plt.show()

# ----------------------------------EXEMPLOS------------------------------------------ #

def exemplo1():
    #função banana ou Rosenbrock com ponto mínimo (a,a**2), a = 1, (1,1)
    def f(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def grad(x):
        return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2),200*(x[1]-x[0]**2)])   
    def hess(x):
        return np.array([[2-400*(-3*x[0]**2 + x[1]),-400*x[0]],[-400*x[0],200]])

    #testa o gradiente (que demora muitas iterações para chegar)
    x,k = gd(f,np.array([-2,2]),grad, plot = True, search=True, itmax=15000)
    print(f"x = {x}")
    print(f"k = {k}")

    #testa o newton
    x,k = newton(f,np.array([-2,2]),grad,hess, plot = True, search=True, itmax=200)
    print(f"x = {x}")
    print(f"k = {k}")

    #testa o fd e o newton em um ponto mais longe do mínimo
    x,k = newton(f,np.array([-20,20]),None,None,search=True, fd=True, plot = True, itmax=200)
    print(f"x = {x}")
    print(f"k = {k}")

    #testa o bfgs
    x,k = bfgs(f,np.array([-2,2]),None,None,fd=True, plot = True, search=True)
    print(f"x = {x}")
    print(f"k = {k}")
    
def exemplo2():
    #é uma função senoide então tem infinitos máximos e mínimos
    def f(x):
        return np.sin(x[0]-x[1])
    def grad(x):
        return np.array([np.cos(x[0]-x[1]),-np.cos(x[0]-x[1])])   
    def hess(x):
        return np.array([[-np.sin(x[0]-x[1]),np.sin(x[0]-x[1])],[np.sin(x[0]-x[1]),-np.sin(x[0]-x[1])]])

    #testa o gradiente
    x,k = gd(f,np.array([-2.73,2.73]),grad, plot = True, search=True)
    print(f"x = {x}")
    print(f"k = {k}")

    #testa o newton, reduzi o tanto de linhas que mostra no gráfico
    global LEVELS
    LEVELS = 15
    x,k = newton(f,np.array([-2.73,2.73]),grad,hess, plot = True, search=True)
    print(f"x = {x}")
    print(f"k = {k}")

    #testa usando diferencas finitas (encontra outro resultado)
    x,k = newton(f,np.array([-2.73,2.73]),None,None,fd=True, plot = True, search=True, itmax=200)
    print(f"x = {x}")
    print(f"k = {k}")

    #teste o bfgs
    x,k = bfgs(f,np.array([-2.73,2.73]),None,None,fd=True, plot = True, search=True)
    print(f"x = {x}")
    print(f"k = {k}")

def exemplo3():
    #função de Easom
    #possui vários pontos críticos, mas o f(pi,pi) = -1 que é o mínimo global
    #porém possui uma área de procura muito pequena em comparação ao todo
    def f(x):
        return -np.cos(x[0])*np.cos(x[1])*np.exp(-1*( (x[0]-np.pi)**2 + (x[1]-np.pi)**2))

    #testa o gd
    x,k = gd(f,np.array([1,1]),None, fd=True, plot = True, search=True)
    print(f"x = {x}")
    print(f"k = {k}")
    
    #testa o newton que, comecando em (1,1) encontra o mínimo global f(pi,pi) = -1
    x,k = newton(f,np.array([1,1]),None,None,fd=True, plot = True, search=True, itmax=200)
    print(f"x = {x}")
    print(f"k = {k}")

    #teste o bfgs
    x,k = bfgs(f,np.array([1,1]),None,None,fd=True, plot = True, search=True)
    print(f"x = {x}")
    print(f"k = {k}")