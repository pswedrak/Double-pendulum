import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import odeint


def model(initial, t, parameters):
    theta1, z1, theta2, z2 = initial
    g, m1, m2, L1, L2 = parameters
    # f = [theta1', theta2', omega1', omega2']
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

g = 10
m1 = 1
m2 = 1
L1 = 0.5
L2 = 0.5
initial = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])
parameters = [g, m1, m2, L1, L2]
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 250

t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

sol = odeint(model, initial, t, args=(parameters,),
              atol=abserr, rtol=relerr)

with open('sol.txt', 'w') as f:
    for t1, w1 in zip(t, sol):
        f.write(str(t1) + ' ' + str(w1[0])+ ' ' +str(w1[1])+ ' ' +str(w1[2])+ ' ' +str(w1[3]) + '\n')

t, theta1dot, z1dot, theta2dot, z2dot = np.loadtxt('sol.txt', unpack=True)

x1 = L1 * np.sin(theta1dot)
y1 = -L1 * np.cos(theta1dot)
x2 = x1 + L2*np.sin(theta2dot)
y2 = y1 - L2*np.cos(theta2dot)

def draw_simulation():
    fig = plt.figure()

    def update(frame):
        plt.clf()
        plt.plot([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        plt.plot([0, x1[frame]], [0, y1[frame]])
        #plt.scatter(x, y) - masses
        #plt.plot(x_log, y_log) - draw saved trajectories
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])

    plot = FuncAnimation(fig, update, frames=range(numpoints), interval=50)
    plt.show()

draw_simulation()