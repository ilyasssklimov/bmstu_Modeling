import math as m
from prettytable import PrettyTable


class CauchyProblem:
    def __init__(self, func):
        self.func = func

    def picard(self, n, h, x, y):
        def f1(a):
            return a ** 3 / 3

        def f2(a):
            return f1(a) + a ** 7 / 63

        def f3(a):
            return f2(a) + (a ** 11) * (2 / 2079) + (a ** 15) / 59535

        def f4(a):
            q1 = (a ** 15) * (2 / 93555)
            q2 = (a ** 19) * (2 / 3393495)
            q3 = (a ** 19) * (2 / 2488563)
            q4 = (a ** 23) * (2 / 86266215)
            q5 = (a ** 23) * (1 / 99411543)
            q6 = (a ** 27) * (2 / 3341878155)
            q7 = (a ** 31) * (1 / 109876902975)
            return f3(a) + q1 + q2 + q3 + q4 + q5 + q6 + q7

        y_out = [[y] for _ in range(4)]

        for i in range(n - 1):
            x += h
            y_out[0].append(f1(x))
            y_out[1].append(f2(x))
            y_out[2].append(f3(x))
            y_out[3].append(f4(x))

        return y_out

    def euler(self, n, h, x, y):
        y_out = []

        for i in range(n):
            y += h * self.func(x, y)
            y_out.append(y)
            x += h

        return y_out

    def runge_kutta(self, n, h, x, y):
        y_out = []
        k = h / 2

        for i in range(n):
            y_out.append(y)
            y += h * self.func(x + k, y + k * self.func(x, y))
            x += h

        return y_out

    def solve_equation(self):
        table = PrettyTable()
        table.field_names = ['x', 'Пикар (1)', 'Пикар (2)', 'Пикар (3)', 'Пикар (4)', 'Эйлер', 'Рунге-Кутта']

        h = 1e-5
        x, y0, end = 0, 0, 2.0
        n = m.ceil((end - x) / h) + 1

        x_arr = [x + h * i for i in range(n)]
        y1 = self.runge_kutta(n, h, x, y0)
        y2 = self.euler(n, h, x, y0)
        y3 = self.picard(n, h, x, y0)

        step = int(n / 100)
        for i in range(0, n, step):
            table.add_row([round(x_arr[i], 8), round(y3[0][i], 8), round(y3[1][i], 8),
                           round(y3[2][i], 8), round(y3[3][i], 8), round(y1[i], 8), round(y2[i], 8)])

        print(table)


def main():
    cauchy = CauchyProblem(lambda x, u: x ** 2 + u ** 2)
    cauchy.solve_equation()


if __name__ == '__main__':
    main()
