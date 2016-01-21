""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np

class Integrator(object):
    """A generic Runge-Kutta 4 differential equation integrator class
    
    If ODE, solve a conventional ODE:
        dy/dt = deriv_fn(t,y)
    
    If TCL or TC, solve a generalized differential equation:
        dy/dt = Omega y(t) + R(t)y(t) + \int_0^t ds K(t-s) y(s)
    following Leathers and Micha, Chem. Phys. Lett. 415, 46 (2005).

    """

    def __init__(self, diffeq_type, dt, deriv_fn=None, Omega=None, R=None, K=None):
        """Initialize the Integrator class.

        Parameters
        ----------
        diffeq_type : str
            The differential equation type, either 'ODE', 'TC', or 'TCL'.
        dt : float
            The integrator timestep.
        deriv_fn : function
            The derivative function dy/dt for an ODE.
        Omega : np.array
            The free-streaming part of a TC or TCL differential equation, given
            as a len(y) x len(y) matrix.
        R : function
            The time-local function which takes two arguments (n,i) and returns
            R(n*dt + c[i]*dt/4).
        K : function
            The time-nonlocal function which takes four arguments (n,i,m,j) and
            returns K(n*dt + c[i]*dt, m*dt + c[j]*dt).

        """
        assert(diffeq_type in ['ODE', 'TCL', 'TC'])

        self.dt = dt
        # Standard RK4 parameters
        self.order = 4
        self.a = np.array([[  0.,   0.,  0.,  0.],
                           [1./2,   0.,  0.,  0.],
                           [  0., 1./2,  0.,  0.],
                           [  0.,   0.,  1.,  0.]])
        self.b = np.array([1./6, 1./3, 1./3, 1./6])
        self.c = np.array([  0., 1./2, 1./2,   1.])

        if diffeq_type == 'ODE':
            self.integrate = self.integrate_ode
            self.deriv = deriv_fn
        elif diffeq_type == 'TC':
            self.integrate = self.integrate_convolution
            self.Omega = Omega
            self.K = K
            self.Y = []
        elif diffeq_type == 'TCL':
            self.integrate = self.integrate_convolutionless
            self.Omega = Omega
            self.R = R
        else:
            # Should never get here due to assertion above.
            pass

    def set_initial_value(self, y0, t0):
        """Set initial value and time of the differential equation's y(t).

        Parameters
        ----------
        y0 : float or np.array
            Initial value of the time-dependent object y.
        t0 : float
            Initial time.

        """
        self.y = y0
        self.t0 = t0
        self.t = t0
        self.n = 0

    def integrate_ode(self): 
        """Advance a regular ODE one time-step."""
        t, dt = self.t, self.dt
        a, b, c = self.a, self.b, self.c

        k = []
        dy = np.zeros_like(self.y)
        for i in range(self.order):
            Yi = self.y.copy()
            #for j in range(i):
            j = i-1
            if j >= 0:
                Yi += dt*a[i,j]*k[j]
            ki = self.deriv(t + c[i]*dt, Yi)
            k.append( ki )

            dy += dt*b[i]*ki

        self.y += dy
        self.t += dt
        self.n += 1

    def integrate_convolutionless(self): 
        """Advance a time-local (convolutionless) diffeq one time-step."""
        n, dt = self.n, self.dt
        a, b, c = self.a, self.b, self.c
        Omega, R = self.Omega, self.R

        k = []
        dy = np.zeros_like(self.y)
        for i in range(self.order):
            Yi = self.y.copy()
            #for j in range(i):
            j = i-1
            if j >= 0:
                Yi += dt*a[i,j]*k[j]
            ki = np.dot(Omega, Yi) + np.dot(R(n,i), Yi)
            k.append( ki )

            dy += dt*b[i]*ki

        self.y += dy
        self.t += dt
        self.n += 1
        
    def integrate_convolution(self): 
        """Advance a time-nonlocal (convolution) diffeq one time-step."""
        n, dt = self.n, self.dt
        a, b, c = self.a, self.b, self.c
        Omega, K = self.Omega, self.K

        def q(m,i):
            qmt = np.zeros_like(self.y)
            for l in range(m):
                for j in range(self.order):
                    qmt += dt*b[j]*np.dot(K(m,i,l,j), self.Y[l][j])
            return qmt

        kn = []
        Yn = []
        dy = np.zeros_like(self.y)
        for i in range(self.order):
            Zni = q(n, i)
            Yni = self.y.copy()
            #for j in range(i):
            j = i-1
            if j >= 0:
                Zni += dt*a[i,j]*np.dot(K(n,i,n,j), Yn[j])
                Yni += dt*a[i,j]*kn[j]
            kni = np.dot(Omega, Yni) + Zni
            kn.append( kni )
            Yn.append( Yni )

            dy += dt*b[i]*kni

        self.Y.append(Yn)
        self.y += dy
        self.t += dt
        self.n += 1
