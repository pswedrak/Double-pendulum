import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import odeint
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QCheckBox, QLineEdit, QLabel
from PyQt5.QtCore import pyqtSlot


def model(initial, t, parameters):
    theta1, z1, theta2, z2 = initial
    g, m1, m2, L1, L2 = parameters
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot


def draw_simulation(x1, x2, y1, y2, L1, L2, numpoints, dt, m1, m2, drawing_params):
    fig = plt.figure()

    def update(frame, draw_lower, draw_upper):
        plt.clf()

        # print time
        plt.title("t = " + str(np.round(dt*frame, 1)) + "s")

        # plot arms
        plt.plot([x1[frame], x2[frame]], [y1[frame], y2[frame]], 'grey', zorder=1)
        plt.plot([0, x1[frame]], [0, y1[frame]], 'grey', zorder=1)
        # plot blobs
        plt.scatter([x1[frame]], [y1[frame]], c='b', s=20*m1, zorder=10)
        plt.scatter([x2[frame]], [y2[frame]], c='g', s=20*m2, zorder=10)

        # plot trajectories
        if draw_lower:
            plt.plot(x2[0:frame], y2[0:frame], 'g', alpha=0.7, zorder=0)
        if draw_upper:
            plt.plot(x1[0:frame], y1[0:frame], 'b', alpha=0.7, zorder=0)

        # set bounds
        plt.ylim([-(L1+L2+0.1), (L1+L2+0.1)])
        plt.xlim([-(L1+L2+0.1), (L1+L2+0.1)])

    draw_lower, draw_upper = drawing_params
    plot = FuncAnimation(fig, update, frames=range(numpoints), fargs=(draw_lower, draw_upper,), interval=200*dt)
    plt.show()


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Double pendulum'
        self.left = 300
        self.top = 300
        self.width = 425
        self.height = 450
        self.textbox1 = QLineEdit(self)   # g
        self.textbox2 = QLineEdit(self)   # m1
        self.textbox3 = QLineEdit(self)   # m2
        self.textbox4 = QLineEdit(self)   # L1
        self.textbox5 = QLineEdit(self)   # L2
        self.textbox6 = QLineEdit(self)   # time
        self.checkbox1 = QCheckBox(self)  # lower trajectory
        self.checkbox2 = QCheckBox(self)  # upper trajectory
        self.textbox7 = QLineEdit(self)   # dt
        self.textbox8 = QLineEdit(self)   # init theta1
        self.textbox9 = QLineEdit(self)   # init theta2
        self.l0 = QLabel(self)   # param label
        self.l1 = QLabel(self)
        self.l2 = QLabel(self)
        self.l3 = QLabel(self)
        self.l4 = QLabel(self)
        self.l5 = QLabel(self)
        self.l6 = QLabel(self)
        self.l7 = QLabel(self)
        self.l8 = QLabel(self)
        self.l9 = QLabel(self)
        self.l10 = QLabel(self)  # warning
        self.l11 = QLabel(self)  # settings label
        self.l12 = QLabel(self)
        self.l13 = QLabel(self)
        self.button = QPushButton('Simulate', self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.l0.move(20, 5)
        self.l0.resize(280, 20)
        self.l0.setText("Input parameters")

        self.l1.move(20, 18)
        self.l1.setText("g: ")
        self.textbox1.move(100, 25)
        self.textbox1.resize(280, 20)
        self.textbox1.setText("10.0")

        self.l2.move(20, 48)
        self.l2.setText("m1 (kg): ")
        self.textbox2.move(100, 55)
        self.textbox2.resize(280, 20)
        self.textbox2.setText("1")

        self.l3.move(20, 78)
        self.l3.setText("m2 (kg): ")
        self.textbox3.move(100, 85)
        self.textbox3.resize(280, 20)
        self.textbox3.setText("2")

        self.l4.move(20, 108)
        self.l4.setText("L1 (m): ")
        self.textbox4.move(100, 115)
        self.textbox4.resize(280, 20)
        self.textbox4.setText("0.5")

        self.l5.move(20, 138)
        self.l5.setText("L2 (m): ")
        self.textbox5.move(100, 145)
        self.textbox5.resize(280, 20)
        self.textbox5.setText("0.8")

        self.l6.move(20, 168)
        self.l6.setText("simulation \ntime (s): ")
        self.textbox6.move(100, 175)
        self.textbox6.resize(280, 20)
        self.textbox6.setText("10.0")

        self.l12.move(20, 198)
        self.l12.setText("initial angle (°) \nof upper mass: ")
        self.textbox8.move(100, 205)
        self.textbox8.resize(280, 20)
        self.textbox8.setText("-90")

        self.l13.move(20, 228)
        self.l13.setText("initial angle (°) \nof lower mass: ")
        self.textbox9.move(100, 235)
        self.textbox9.resize(280, 20)
        self.textbox9.setText("-90")

        self.l7.move(20, 290)
        self.l7.setText("Draw lower mass \ntrajectory")
        self.checkbox1.move(120, 290)
        self.checkbox1.setChecked(True)

        self.l8.move(20, 323)
        self.l8.setText("Draw upper mass \ntrajectory")
        self.checkbox2.move(120, 318)
        self.checkbox2.setChecked(False)

        self.l9.move(20, 348)
        self.l9.setText("dt (s): ")
        self.textbox7.move(100, 355)
        self.textbox7.resize(280, 20)
        self.textbox7.setText("0.05")

        self.l10.move(20, 370)
        self.l10.resize(280, 40)
        self.l10.setText("WARNING: changing the value of the time step may affect \nmodel's stability. Use carefully.")

        self.l11.move(20, 270)
        self.l11.resize(280, 20)
        self.l11.setText("Simulation settings")

        self.button.move(20, 410)

        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        dt = float(self.textbox7.text())
        g = float(self.textbox1.text())
        m1 = float(self.textbox2.text())
        m2 = float(self.textbox3.text())
        L1 = float(self.textbox4.text())
        L2 = float(self.textbox5.text())
        t1 = np.pi * float(self.textbox8.text()) / 180
        t2 = np.pi * float(self.textbox9.text()) / 180
        initial = np.array([t1, 0, t2, 0])
        parameters = [g, m1, m2, L1, L2]
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = float(self.textbox6.text())
        numpoints = round(stoptime / dt)

        t = np.linspace(0, stoptime, numpoints)

        sol = odeint(model, initial, t, args=(parameters,),
                     atol=abserr, rtol=relerr)

        with open('sol.txt', 'w') as f:
            for t1, w1 in zip(t, sol):
                f.write(str(t1) + ' ' + str(w1[0]) + ' ' + str(w1[1]) + ' ' + str(w1[2]) + ' ' + str(w1[3]) + '\n')

        t, theta1dot, z1dot, theta2dot, z2dot = np.loadtxt('sol.txt', unpack=True)

        x1 = L1 * np.sin(theta1dot)
        y1 = -L1 * np.cos(theta1dot)
        x2 = x1 + L2 * np.sin(theta2dot)
        y2 = y1 - L2 * np.cos(theta2dot)

        drawing_params = (self.checkbox1.checkState(), self.checkbox2.checkState())
        draw_simulation(x1, x2, y1, y2, L1, L2, numpoints, dt, m1, m2, drawing_params)


def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
