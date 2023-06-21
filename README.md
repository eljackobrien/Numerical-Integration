# Numerical-Integration
Some functions for numerical integration

Functions implemented are the simple Euler method, the Heun[3] (modified Euler) method,
Adams-Bashforth[1] 3rd through 5th order methods and two 4th order Runge-Kutta[2] methods.
These are all explicit methods.

Tested by integrating dy/dx = y (which is the exponential function y = e^x)

Timings for all n=10000 and dx=0.001, averaged over 1000 runs each.
Slope function was simply returning y in this case, not very intensive.
Euler:6ms,    Heun:19ms,    AB3:19ms,    AB4:23ms,    AB5:27ms,    RK4:35ms,    RK4_38:40ms

Repeated but this time performing 1000 additions in the slope function to slow it down.
Averaged over 100 runs.
Euler:156ms,    Heun:469ms,    AB3:170ms,    AB4:174ms,    AB5:178ms,    RK4:635ms,    RK4_38:640ms

Now we see the timings are essentially just proportional to the number of slope calculations per step.

When the step sizes are adjusted so that the errors are similar, then the RK algorithms are
faster since they can use larger step sizes for equivalent accuracy.
The AB algorithms have possibly the best trend in error after many steps but the initial error
from the lower order approximations is carried through.

Of course, this all depends on the function in question. Never use simple Euler though...

[1] https://en.wikipedia.org/w/index.php?title=Linear_multistep_method&oldid=1136115186
[2] https://en.wikipedia.org/w/index.php?title=Runge%E2%80%93Kutta_methods&oldid=1151548276
[3] https://en.wikipedia.org/w/index.php?title=Heun%27s_method&oldid=1159010645
