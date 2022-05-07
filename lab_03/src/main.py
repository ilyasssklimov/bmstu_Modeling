from math import exp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from typing import Callable


def frange(start, end, step):
    str_step = str(step)
    accuracy = len(str_step[str_step.find('.') + 1:])

    i = start
    while i < end:
        yield round(i, accuracy)
        i += step


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

        self.res_z = None
        self.res_up = None
        self.res_u = None
        self.res_f = None
        self.res_div_f = None

    def k(self, z):
        return self.k0 * (self.T(z) / 300) ** 2

    def up(self, z):
        return 3.084 * 1e-4 / (exp(4.799 * 1e4 / self.T(z)) - 1)

    def T(self, z):
        return (self.Tw - self.T0) * pow(z, self.p) + self.T0

    def K(self, z):
        return self.c / (3 * self.k(z))

    def X(self, z, h):
        return (self.K(z) + self.K(z + h)) / 2

    def P(self, z):
        return self.c * self.k(z)

    def F(self, z):
        return -self.c * self.k(z) * self.up(z)

    @staticmethod
    def solve_matrix(matrix: list[tuple[float, float, float]], vector: list[float]):
        n = len(matrix) - 1
        ksi = [0, -matrix[0][1] / matrix[0][0]]
        eta = [0, vector[0] / matrix[0][0]]

        for i in range(1, n):
            a, b, c = matrix[i][0], -matrix[i][1], matrix[i][2]
            ksi.append(c / (b - a * ksi[i]))
            eta.append((-vector[i] + a * eta[i]) / (b - a * ksi[i]))

        result = [(vector[n] - matrix[n][1] * eta[n]) / (matrix[n][1] * ksi[n] + matrix[n][2])]
        for i in range(n):
            result.append(result[i] * ksi[n - i] + eta[n - i])
        result.reverse()

        return result

    def difference_method(self, nrange: tuple[float, float, float]):
        z, end, h = nrange[0], nrange[1], nrange[2]

        tmp = 1 / self.R ** 2 / h * self.X(z, h) * (z + h / 2)
        matrix = [(
            -tmp - h / 4 * (z + h / 4) * (h / 2) * (self.P(z) + 1 / 2 * self.P(z + h / 2)),
            tmp - h / 8 * (z + h / 4) * (h / 2) * self.P(z + h / 2),
            0
        )]
        result = [(self.F(z) + self.F(z + h / 2)) * h * (z + h / 4) * (h / 2) / 4]

        for z in frange(z + h, end - h / 2, h):
            A = (z - h / 2) * self.X(z - h, h) / (self.R ** 2 * h)
            C = (z + h / 2) * self.X(z, h) / (self.R ** 2 * h)

            matrix.append((A, -(A + C + self.P(z) * z * h), C))
            result.append(self.F(z) * z * h)

        tmp = (z - h / 2) * self.X(z - h, h) / (self.R ** 2 * h)
        matrix.append((
            0,
            tmp - h * (z - h / 4) * (h / 2) * self.P(z - h / 2) / 8,
            -tmp - z * 0.393 * self.c / self.R - h / 4 * (z - h / 4) * (h / 2) * (self.P(z - h / 2) / 2 + self.P(z))
        ))
        result.append(h / 4 * (self.F(z - h / 2.) + self.F(z)) * (z - h / 4) * (h / 2))

        return matrix, result

    def solve(self, nrange: tuple[float, float, float]):
        matrix, vector = self.difference_method(nrange)
        return self.solve_matrix(matrix, vector)

    def get_derivative(self, u: list[float], nrange: tuple[float, float, float]):
        result = [0]
        result_2 = [0]
        z, step = nrange[0], nrange[2]

        for i in range(1, len(u) - 1):
            tmp = -self.c / 3 / self.R / self.k(z)
            result.append(tmp * ((u[i + 1] - u[i - 1]) / 2 / step))
            result_2.append(tmp * (u[i + 1] + u[i - 1] - 2 * u[i]) / step ** 2)
            z += step

        result.append(-self.c / 3 / self.R / self.k(z) *
                      (- 4 * u[len(u) - 2] + u[len(u) - 3] + 3 * u[len(u) - 1]) / 2 / step)
        result_2.append(result_2[-1])
        result_2[0] = result_2[1]

        return result, result_2

    def get_result(self, result: list[list[float]], filename: str = None):
        table = PrettyTable()
        table.field_names = ['Z', 'Up', 'U', 'F', 'Div F']
        self.res_z = result[0]
        self.res_up = [self.up(z) for z in result[0]]
        self.res_u = result[1]
        self.res_f = result[2]
        self.res_div_f = result[3]

        for i in range(len(result[0])):
            row = f'{result[0][i]} {self.up(result[0][i]):.6g} {result[1][i]:.6g} {result[2][i]:.6f} {result[3][i]:.6f}'
            table.add_row(row.split())

        if filename:
            with open(filename, 'w') as f:
                f.write(str(table))
        else:
            print(table)

    @staticmethod
    def get_simpson(nrange: tuple[float, float, float], f: Callable):
        a, b, n = nrange[0], nrange[1], (nrange[1] - nrange[0]) / nrange[2]
        width = (b - a) / (n - 1)
        integral = 0

        for i in frange(0, (n - 1) // 2, 1):
            x1 = a + 2 * i * width
            x2 = a + 2 * (i + 1) * width
            integral += f(x1) + 4 * f((x1 + x2) / 2) + f(x2)

        return integral * width / 3

    def integrate(self, u: list[float], nrange: tuple[float, float, float]):
        simpson = self.c * self.R * self.get_simpson(nrange,
                                                     lambda z: self.k(z) * (self.up(z) - u[int(z / nrange[2])]) * z)
        print(f'Result of integration: {simpson:.4f}')

    def create_graphics(self):
        plt.figure(1)
        plt.plot(self.res_z, self.res_up, label='Up')
        plt.plot(self.res_z, self.res_u, label='U')
        plt.xlabel('z')
        plt.ylabel('U')
        plt.legend()

        plt.figure(2)
        plt.plot(self.res_z, self.res_f, label='F')
        plt.xlabel('z')
        plt.ylabel('F')
        plt.legend()

        plt.figure(3)
        plt.plot(self.res_z, self.res_div_f, label='div F')
        plt.xlabel('z')
        plt.ylabel('div F')
        plt.legend()

        plt.show()


def main():
    solution = Solution()

    nrange = (0, 1, 1e-4)
    z = [i for i in frange(*nrange)]

    u = solution.solve(nrange)
    F = solution.get_derivative(u, nrange)

    solution.get_result([z, u, F[0], F[1]], '../data/result.txt')
    solution.integrate(u, nrange)

    solution.create_graphics()


if __name__ == '__main__':
    main()
