import numpy as np

def phi(x):
    if x >= 0 and x <= 10:
        y=np.exp(-0.4527*(x**0.859)+0.0218)
    elif x >= 10:
        y=np.sqrt(np.pi/x)*np.exp(-x/4)*(1-(10/(7*x)))
    else:
        print('x is not a positive number, so the function: phi can not pass !')
        quit()
    return y

def phi_derivative(x):
    if x >= 0 and x <= 10:
        dx = -0.4527 * 0.86 * (x ** (-0.14)) * phi(x)
    else:
        dx = np.sqrt(np.pi) * np.exp(-x / 4) * ((15 / 7) * (x ** (-5 / 2)) - (1 / 7) * (x ** (-3 / 2)) - (1 / 4) * (x ** (-1 / 2)))

    return dx

def phi_inverse(x):
    #用的是Newton's method，也就是牛顿迭代法求零点
    if x >= 0.0388 and x <= 1.0221:
        y= ((0.0218-np.log(x))/0.4527)**(1/0.86)
    else:
        x0=0.0388
        x1=x0-((phi(x0)-x)/phi_derivative(x0))
        delta=np.abs(x1-x0)
        gap=1e-3

        while delta >= gap:
            x0=x1
            x1=x1-((phi(x1)-x)/phi_derivative(x1))
            if x1 > 1e2:
                gap = 10
            delta=np.abs(x1-x0)
        y=x1
    return y




