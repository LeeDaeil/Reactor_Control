from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

DeltaKcr = 0

def PointKinetic(t, x):
    """
    ref. "원자로 보호 ㆍ 감시 및 제어", 나만균, 이기복, 김경석 저, 2014년, pp 353

    :param x:
                x[0] = 원자로 상대출력
                x[1]~x[6] = 6개 지발중성자 모핵종 농도
                x[7] = 핵연료 온도
                x[8] = 감속재 온도
    :param DeltaKcr:
                제어봉 이동으로 인한 반응도
    :return: xdot =
    """
    # 핵연료 온도계수에 의한 반응도 변화
    DeltaKf = -3.1 * 1e-6 * x[7] + 0.0031 * 1e-6
    # 감속재 온도계수에 의한 반응도 변화
    DeltaKm = - 0.6 * 1e-6 * x[8] + 0.0003 * 1e-6
    #
    global DeltaKcr

    DeltaK = DeltaKcr + DeltaKf + DeltaKm
    xdot = [
        (DeltaK / 0.001 - 6.502) * x[0] + 0.0124 * x[1] + 0.0305 * x[2] + 0.111 * x[3] + 0.301 * x[4] + 1.14 * x[5] + 3.01 * x[6],  # x(0) 중성자속
        0.215000 * x[0] - 0.0124 * x[1],  # 1군 지발중성자 모핵종 농도
        1.424000 * x[0] - 0.0305 * x[2],  # 2군 지발중성자 모핵종 농도
        1.274000 * x[0] - 0.1110 * x[3],  # 3군 지발중성자 모핵종 농도
        2.568000 * x[0] - 0.3010 * x[4],  # 4군 지발중성자 모핵종 농도
        0.748000 * x[0] - 1.1400 * x[5],  # 5군 지발중성자 모핵종 농도
        0.273000 * x[0] - 3.0100 * x[6],  # 6군 지발중성자 모핵종 농도
        0.000200 * x[0] - 0.2000 * x[7],  # 핵연료 온도
        0.000005 * x[0] - 0.0100 * x[8],  # 감속재 온도
    ]
    return xdot

x0 = [1, 17.3387, 46.6885, 11.4775, 8.5316, 0.6561, 0.0907, 0.001, 0.0005]
t0 = 0


set_y = [1 for _ in range(100)] + np.linspace(1, 0.5, 100).tolist() + [0.5 for _ in range(300)] + np.linspace(0.5, 1, 100).tolist() + [1 for _ in range(100)]

r = integrate.ode(PointKinetic).set_integrator("dopri5")
r.set_initial_value(x0, t0)

power_list = []
dt = 1

# PID Controller
Kp, Ki, Kd = 0.002, 0.0003, 0.0001
err, err1, err2 = 0, 0, 0

for t in range(len(set_y)):
    o = r.integrate([t, t+dt])
    err2 = err1
    err1 = err
    err = set_y[t] - o[0]
    DeltaKcr = DeltaKcr + Kp * (err - err1) + Ki * err + Kd *((err - err1) - (err1 - err2))
    power_list.append(o[0])

plt.plot(power_list, 'black')
plt.plot(set_y)
plt.savefig('power.png', dpi=600)
plt.show()





