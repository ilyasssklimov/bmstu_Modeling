from math import exp
import matplotlib.pyplot as plt
from matplotlib import ticker


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


class Solution:
    def __init__(self):
        self.k0 = 0.0008
        self.Tw = 2000
        self.T0 = 1e4
        self.p = 4
        self.R = 0.35
        self.c = 3e10

        self.z0 = 0
        self.z1 = 1
        self.F0 = 0

    def k(self, z):
        return self.k0 * (self.T(z) / 300) ** 2

    def up(self, z):
        return 3.084 * 1e-4 / (exp(4.799 * 1e4 / self.T(z)) - 1)

    def T(self, z):
        return (self.Tw - self.T0) * pow(z, self.p) + self.T0

    def du(self, z, _u, f):
        return -f * 3 * self.R * self.k(z) / self.c

    def df(self, z, u, f):
        tmp = self.R * self.c * self.k(z) * (self.up(z) - u)
        if z != 0:
            return tmp - f / z
        else:
            return tmp / 2

    def runge_kutta(self, u0):
        h = 1e-4
        half_h = h / 2
        z, u, f = self.z0, u0, self.F0
        z_, u_, f_ = [z], [u], [f]

        while z <= self.z1:
            k1 = h * self.du(z, u, f)
            q1 = h * self.df(z, u, f)

            k2 = h * self.du(z + half_h, u + k1 / 2, f + q1 / 2)
            q2 = h * self.df(z + half_h, u + k1 / 2, f + q1 / 2)

            k3 = h * self.du(z + half_h, u + k2 / 2, f + q2 / 2)
            q3 = h * self.df(z + half_h, u + k2 / 2, f + q2 / 2)

            k4 = h * self.du(z + h, u + k3, f + q3)
            q4 = h * self.df(z + h, u + k3, f + q3)

            u += (k1 + k2 * 2 + k3 * 2 + k4) / 6
            f += (q1 + q2 * 2 + q3 * 2 + q4) / 6
            z += h

            u_.append(u)
            f_.append(f)
            z_.append(z)

        return u_, f_, z_

    def half_division(self, left, right, grad):
        middle = (left + right) / 2
        if abs(left - right) < 1e-8:
            return middle

        result = self.diff(middle)
        if grad:
            if result > 0:
                return self.half_division(left, middle, grad)
            elif result < 0:
                return self.half_division(middle, right, grad)
        else:
            if result > 0:
                return self.half_division(middle, right, grad)
            elif result < 0:
                return self.half_division(left, middle, grad)

        return middle

    def diff(self, u0):
        runge = self.runge_kutta(u0)
        u1, f1 = runge[0][-1], runge[1][-1]
        return f1 - 0.393 * self.c * u1

    def shoot(self):
        h = 0.01
        ksi, z0 = 0, self.z0
        u0 = ksi * self.up(z0)

        sign_0 = sign(self.diff(u0))
        sign_cur = sign_0

        while sign_cur == sign_0:
            ksi += h
            u0 = ksi * self.up(z0)
            sign_cur = sign(self.diff(u0))

        ksi -= h
        u0_left = ksi * self.up(z0)
        u0_right = u0

        grad = True if sign_0 == -1 else False
        return self.half_division(u0_left, u0_right, grad)

    @staticmethod
    def create_main_graphics(z, f, u, up, fig=1):
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))

        plt.figure(fig)
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(z, f, label="f(z)")
        plt.legend()

        ax2 = plt.subplot(1, 2, 2)
        plt.plot(z, u, label="u(z)")
        plt.plot(z, up, label="up(z)")
        plt.legend()

        ax2.yaxis.set_major_formatter(formatter)
        plt.sca(ax1)
        ax1.yaxis.set_major_formatter(formatter)

        plt.show()

    def changing_k0(self, start, end):
        step = (end - start) / 10
        self.k0 = start
        k_, u_ = [], []

        for _ in range(10):
            k_.append(self.k0)
            u_.append(self.shoot())
            self.k0 += step

        self.k0 = 0.0008
        return k_, u_

    def changing_Tw(self, start, end):
        step = (end - start) / 8
        self.Tw = start

        Tw_ = []
        u_ = []
        for _ in range(8):
            Tw_.append(self.Tw)
            u_.append(self.shoot())
            self.Tw += step

        self.Tw = 2000
        return Tw_, u_

    def changing_T0(self, start, end):
        step = (end - start) / 10
        self.T0 = start

        T0_ = []
        u_ = []
        for _ in range(10):
            T0_.append(self.T0)
            u_.append(self.shoot())
            self.T0 += step

        self.T0 = 1e4
        return T0_, u_

    def changing_p(self, start, end):
        step = (end - start) / 11
        self.p = start

        p_ = []
        u_ = []
        for _ in range(11):
            p_.append(self.p)
            u_.append(self.shoot())
            self.p += step

        self.p = 4
        return p_, u_

    def changing_R(self, start, end):
        step = (end - start) / 10
        self.R = start

        R_ = []
        u_ = []
        for _ in range(10):
            R_.append(self.R)
            u_.append(self.shoot())
            self.R += step

        self.R = 0.35
        return R_, u_

    @staticmethod
    def create_extra_graphic(data, label, figure):
        plt.figure(figure)
        plt.plot(data[0], data[1], label=label)
        plt.legend()

    def create_extra_graphics(self, fig=1):
        self.create_extra_graphic(self.changing_k0(0.0004, 0.0014), 'u0(k0)', fig + 1)
        print('Graphic u0(k0) is created')
        self.create_extra_graphic(self.changing_Tw(500, 4500), 'u0(Tw)', fig + 2)
        print('Graphic u0(Tw) is created')
        self.create_extra_graphic(self.changing_T0(5000, 15000), 'u0(T0)', fig + 3)
        print('Graphic u0(T0) is created')
        self.create_extra_graphic(self.changing_p(4, 15), 'u0(p)', fig + 4)
        print('Graphic u0(p) is created')
        self.create_extra_graphic(self.changing_R(0.1, 11), 'u0(R)', 5)
        print('Graphic u0(R) is created')

        plt.show()


def main():
    solution = Solution()
    u0 = solution.shoot()

    u, f, z = solution.runge_kutta(u0)
    up = [solution.up(z_cur) for z_cur in z]

    solution.create_main_graphics(z, f, u, up)
    solution.create_extra_graphics()


if __name__ == '__main__':
    main()
