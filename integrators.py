# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:19:26 2023

@author: OBRIEJ25
"""
import numpy as np

def euler_propagate(slope, y0, x0, n, dx, prog_bar=False):
    y = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0

    for i in range(0, n-1):
        y[i+1] = y[i] + slope(y[i], x[i]) * dx

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")
    return x, y

def heun_propagate(slope, y0, x0, n, dx, prog_bar=False):
    y = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0
    for i in range(0, n-1):
        y[i+1] = y[i] + slope(y[i], x[i]) * dx
        y[i+1] = y[i] + (slope(y[i], x[i]) + slope(y[i+1], x[i+1])) * dx/2

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")

    return x, y

def ab3_propagate(slope, y0, x0, n, dx, prog_bar=False):
    y = np.zeros((n, *np.squeeze([y0]).shape))
    dy = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0

    # Store slopes to prevent recalculation every step, storing all slopes for simplicity
    dy[0] = slope(y[0], x[0])
    y[1] = y[0] + dy[0]*dx
    dy[1] = slope(y[1], x[1])
    y[2] = y[1] + (1.5*dy[1] - 0.5*dy[0])*dx

    for i in range(2, n-1):
        dy[i] = slope(y[i], x[i])
        y[i+1] = y[i] + (23*dy[i] - 16*dy[i-1] + 5*dy[i-2])*dx/12

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")

    return x, y

def ab4_propagate(slope, y0, x0, n, dx, prog_bar=False):
    y = np.zeros((n, *np.squeeze([y0]).shape))
    dy = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0

    # Store slopes to prevent recalculation every step, storing all slopes for simplicity
    dy[0] = slope(y[0], x[0])
    y[1] = y[0] + dy[0]*dx
    dy[1] = slope(y[1], x[1])
    y[2] = y[1] + (1.5*dy[1] - 0.5*dy[0])*dx
    dy[2] = slope(y[2], x[2])
    y[3] = y[2] + (23*dy[2] - 16*dy[1] + 5*dy[0])*dx/12

    for i in range(3, n-1):
        dy[i] = slope(y[i], x[i])
        y[i+1] = y[i] + (55*dy[i] - 59*dy[i-1] + 37*dy[i-2] - 9*dy[i-3])*dx/24

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")

    return x, y

def ab5_propagate(slope, y0, x0, n, dx, prog_bar=False):
    y = np.zeros((n, *np.squeeze([y0]).shape))
    dy = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0

    # Store slopes to prevent recalculation every step, storing all slopes for simplicity
    dy[0] = slope(y[0], x[0])
    y[1] = y[0] + dy[0]*dx
    dy[1] = slope(y[1], x[1])
    y[2] = y[1] + (1.5*dy[1] - 0.5*dy[0])*dx
    dy[2] = slope(y[2], x[2])
    y[3] = y[2] + (23*dy[2] - 16*dy[1] + 5*dy[0])*dx/12
    dy[3] = slope(y[3], x[3])
    y[4] = y[3] + (55*dy[3] - 59*dy[2] + 37*dy[1] - 9*dy[0])*dx/24

    for i in range(4, n-1):
        dy[i] = slope(y[i], x[i])
        y[i+1] = y[i] + (1901*dy[i] - 2774*dy[i-1] + 2616*dy[i-2] - 1274*dy[i-3] + 251*dy[i-4])*dx/720

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")

    return x, y

def rk4_propagate(slope, y0, x0, n, dx, prog_bar=False):
    y = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0

    # Runge-Kutta method has intermediate steps rather than storing previous ones
    # More calculations per step so slower but for the same error, larger steps can be taken
    for i in range(0, n-1):
        dy0 = slope(y[i], x[i])
        dy1 = slope(y[i] + dy0*dx/2,  x[i] + dx/2)
        dy2 = slope(y[i] + dy1*dx/2,  x[i] + dx/2)
        dy3 = slope(y[i] + dy2*dx,  x[i] + dx)
        y[i+1] = y[i] + (dy0 + 2*dy1 + 2*dy2 + dy3)*dx/6

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")

    return x, y

def rk4_38_propagate(slope, y0, x0, n, dx, prog_bar=False):
    # The slightly more accurate and less efficient "3/8" version of O(4) RK
    y = np.zeros((n, *np.squeeze([y0]).shape))
    x = np.arange(n)*dx + x0
    y[0] = y0

    for i in range(0, n-1):
        dy0 = slope(y[i], x[i])
        dy1 = slope(y[i] + dy0*dx/3,  x[i] + dx/3)
        dy2 = slope(y[i] + dy1*dx - dy0*dx/3,  x[i] + 2*dx/3)
        dy3 = slope(y[i] + dy2*dx - dy1*dx + dy0*dx,  x[i] + dx)
        y[i+1] = y[i] + (dy0 + 3*dy1 + 3*dy2 + dy3)*dx/8

        # prog bar 50 chars long, print formatted string after checking whether to update and print
        if not i%(n//50) and prog_bar:
            print(f"\r[{int(i*50/n)*'#':50.50s}] {100*i/n:.0f}%", end='')
    if prog_bar: print("")

    return x, y


#%% Testing using an exponential, y = e^x  ->  dy/dx = y
if '__main__' in __name__:
    from timeit import default_timer as timer
    import matplotlib.pyplot as plt
    def dydx(y, x):
        #for i in range(1000): 1+1
        return y

    T0,T1,T2,T3,T4,T5,T6,T7 = 0,0,0,0,0,0,0,0
    num_runs = 1

    # Running functions one after another and adding the time taken to a total variable.
    # This helps to reduce the effect of system performance drifting over time.
    for i in range(num_runs):
        t0 = timer()
        x_0, y_0 = euler_propagate(dydx, 1, 0, 1000000, 0.00001, False)
        t1 = timer()
        x_1, y_1 = heun_propagate(dydx, 1, 0, 200000, 0.00005, False)
        t2 = timer()
        x_2, y_2 = ab3_propagate(dydx, 1, 0, 100000, 0.0001, False)
        t3 = timer()
        x_3, y_3 = ab4_propagate(dydx, 1, 0, 100000, 0.0001, False)
        t4 = timer()
        x_4, y_4 = ab5_propagate(dydx, 1, 0, 100000, 0.0001, False)
        t5 = timer()
        x_5, y_5 = rk4_propagate(dydx, 1, 0, 640, 0.015625, False)
        t6 = timer()
        x_6, y_6 = rk4_38_propagate(dydx, 1, 0, 640, 0.015625, False)
        t7 = timer()

        T0 += (t1 - t0)
        T1 += (t2 - t1)
        T2 += (t3 - t2)
        T3 += (t4 - t3)
        T4 += (t5 - t4)
        T5 += (t6 - t5)
        T6 += (t7 - t6)

    names = ['Euler','Heun','AB3','AB4','AB5','RK4','RK4_38']
    times = [T0,T1,T2,T3,T4,T5,T6]

    for name,time in zip(names,times):
        print(f"{name:7.7s} algorithm took {1e3*time/num_runs:4.0f}ms per run over {num_runs} runs")

    """
    Timings for all n=10000 and dx=0.001, averaged over 1000 runs each.
    Slope function was simply returning y in this case, not very intensive.
    Euler:6ms,    Heun:19ms,    AB3:19ms,    AB4:23ms,    AB5:27ms,    RK4:35ms,    RK4_38:40ms

    Repeated but this time performing 1000 additions in the slope function to slow it down.
    Averaged over 100 runs.
    Euler:156ms,    Heun:469ms,    AB3:170ms,    AB4:174ms,    AB5:178ms,    RK4:635ms,    RK4_38:640ms

    Now we see the timings are essentially just proportional to the number of slope calculations per step.

    When the step sizes are adjusted so that the errors are similar, then the RK algorithms are FAR
    faster since they can use much larger step sizes for equivalent accuracy.
    The AB algorithms have possibly the best trend in error after many steps but the initial error
    from the lower order approximations is carried through.

    Of course, this all depends on the function in question. Never use simple Euler though...
    """

    # Plot error evolution
    plt.figure(figsize=[6.5,4], constrained_layout=True)
    xs, ys = [x_0,x_1,x_2,x_3,x_4,x_5,x_6], [y_0,y_1,y_2,y_3,y_4,y_5,y_6]
    for i,[x,y,name] in enumerate(zip(xs, ys, names)):
        plt.plot(x+0.1*i, abs(y - np.e**x), label=name)
    plt.yscale('log')
    plt.legend(fontsize=12, ncol=2)
    plt.title("Absolute Error of Integrators for $e^x$", fontsize=14)
    plt.xlabel('$x$', fontsize=12), plt.ylabel("$y_{err} = y - e^x$", fontsize=12)
    plt.show()


    # Testing for 2D inputs
    y0 = np.random.randint(0,5,[3])
    test = euler_propagate(dydx, y0, 0, 1000, 0.001, False)[1]
    assert test[-1].shape == y0.shape

    y0 = np.random.randint(0,5,[2,4])
    test = ab3_propagate(dydx, y0, 0, 1000, 0.001, False)[1]
    assert test[-1].shape == y0.shape

    y0 = np.random.randint(0,5,[2,5,7])
    test = rk4_38_propagate(dydx, y0, 0, 1000, 0.001, False)[1]
    assert test[-1].shape == y0.shape



