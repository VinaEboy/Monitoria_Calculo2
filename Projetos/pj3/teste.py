import projeto3 as pj
import numpy as np

def exemplo1():
    def f(x):
        return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2
    
    def grad(x):
        return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])
    
    x,k = pj.gd(f,np.array([0,0]),grad, plot=True)
    
    print(f"x = {x}")
    print(f"k = {k}")

def exemplo2():
    def f(x):
        return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2
    
    def grad(x):
        return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])
    
    def hess(x):
        return np.array([[12*x[0]**2-4,-1],[-1,2]])

    pj.LEVELS = 30
    x,k = pj.gd(f,np.array([5,5]),grad,1e-6,1e-2,plot=True)
    print(f"x = {x}")
    print(f"k = {k}")

    x,k = pj.newton(f,np.array([5,5]),grad,hess,eps=1e-6,alpha=1e0,plot=True)
    print(f"x = {x}")
    print(f"k = {k}")

def exemplo3():

    def f(x):
        return np.sin(x[0]*x[1])*np.cos(x[1]**2)
    
    g = pj.fin_diff(f,np.array([-1,2]),1,1e-5)
    print(f"∇f(x) = {g}")
    H = pj.fin_diff(f,np.array([-1,2]),2,1e-5)
    print(f"∇2f(x) = {H}")

    def f(x):
        return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2
    
    x,k = pj.gd(f,np.array([3,3]),None,eps=1e-8,alpha=1e-2,fd =True,plot=True)
    print(f"x = {x}")
    print(f"k = {k}")

def exemplo4():
    def f(x):
        return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2
    
    def grad(x):
        return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])

    x,k = pj.gd(f,np.array([10,10]),grad,eps=1e-8,alpha=1e-2,itmax=100000,plot=True)
    print(f"x = {x}")
    print(f"k = {k}")

    x,k = pj.gd(f,np.array([10,10]),grad,eps=1e-8,alpha=1e-3,itmax=100000,plot=True)
    print(f"x = {x}")
    print(f"k = {k}")

    x,k = pj.gd(f,np.array([10,10]),grad,eps=1e-8,itmax=100000,plot=True,search=True)
    print(f"x = {x}")
    print(f"k = {k}")

#o exemplo 5 demora mesmo
def exemplo5():
    def f(x):
        sum = 0
        for i in range(len(x)-1):
            sum += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
        return sum
    
    def grad(x):
        return pj.fin_diff(f,x,1,1e-7)
    
    x,k = pj.gd(f,np.zeros(10),None,eps=1e-5,itmax=100000,fd=True,search=True)
    print(f"x = {x}")
    print(f"k = {k}")
    print(f"f(x) = {f(x):e}")
    print(f"||∇ f(x)|| = {np.linalg.norm(grad(x)):e}")

def exemplo6():
    pj.exemplo1()  
    pj.exemplo2()
    pj.exemplo3()

def todos_exemplos():
    exemplo1()
    exemplo2()
    exemplo3()
    exemplo4()
    exemplo5()
    exemplo6()

todos_exemplos()
