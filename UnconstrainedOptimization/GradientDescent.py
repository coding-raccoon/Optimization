"""
演示坐标下降法解决一个凸函数的最优化问题的例子
优化函数：
        f(x,y)=x**2+3*x**26x*y-6
"""
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cvx_function(X, Y):
    """
    凸函数函数值的计算
    """
    z1 = X**2
    z2 = Y**2
    z = z1 + 3 * z2 - 2 * X * Y - 6
    return z


def cvx_fucntion_gradient(x, y):
    """
    计算凸函数的梯度
    :param X:
    :param Y:
    :return:
    """
    grad_x = 2 * x - 2 * y
    grad_y = 6 * y - 2 * x
    return grad_x, grad_y


def rosenbrock(X, Y):
    """
    非凸函数rosenbrock函数值计算
    """
    z1 = (1.0 - X)**2
    z2 = 100 * (Y - X*X)**2
    z = z1 + z2
    return z


def rosenbrock_gradient(x, y):
    """
    计算非凸函数rosenbrock的梯度
    """
    grad_x = 400 * x * x * x + 2 * x - 400 * x * y - 2
    grad_y = 200 * (y - x**2)
    return grad_x, grad_y


def generate_grid(x_1, x_2, y_1, y_2, delta, f):
    """
    生成二维网格，并计算网格中各个点的值，用于后续画登高线三维图
    :param x_1: x最小值
    :param x_2: x最大值
    :param y_1: y最小值
    :param y_2: y最大值
    :param delta: 网格中各点间隔
    :param f: 函数
    :return: 网格坐标以及网格中各点函数值
    """
    x = np.arange(x_1, x_2, delta)
    y = np.arange(y_1, y_2, delta)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    return X, Y, Z


def plot_2D_figure(X, Y, Z, x, y, filepath):
    """
    画二维图
    :param X: 网格x坐标
    :param Y: 网格y坐标
    :param Z: 网格坐标各个点函数值
    :param x:优化点列x坐标变化
    :param y:优化点列y坐标变化
    """
    plt.figure()
    plt.contourf(X, Y, Z, 10)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    plt.plot(x, y, c='r')
    plt.savefig(filepath)
    plt.show()


def plot_3D_figure(X, Y, Z, x, y, z, filepath):
    """
    画三维图
    :param X:网格x坐标
    :param Y: 网格y坐标
    :param Z: 网格中各点函数值
    :param x: 优化点列x坐标变化
    :param y: 优化点点列y坐标变化
    :param z: 优化点列函数值变化
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x, y, z, c='r', linewidth=2)
    plt.colorbar(p, shrink=0.8)
    plt.savefig(filepath)
    plt.show()
    pass


def generate_points(x_start, y_start, f, grad, alpha, steps):
    """
       根据坐标下降法，生成优化过程中的点列
    :param grad: 梯度计算函数
    :param x_start: 起始点x坐标
    :param y_start: 起始点y坐标
    :param f: 函数
    :param alpha: 学习率
    :param steps: 迭代的步数
    :return: 优化点列中各点x，y坐标以及对应的函数值
    """
    X, Y, Z = [x_start], [y_start], [f(x_start, y_start)]
    for i in range(1, steps):
        grad_x, grad_y = grad(X[i-1], Y[i-1])
        x_new, y_new = X[i-1] - alpha * grad_x, Y[i-1] - alpha * grad_y
        Z.append(f(x_new, y_new))
        X.append(x_new)
        Y.append(y_new)
    return X, Y, Z


if __name__ == "__main__":
    # x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    # x_start, y_start = 3.4, 3.5
    # X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    # x, y, z = generate_points(x_start, y_start, cvx_function, cvx_fucntion_gradient, 0.05, 100)
    # plot_2D_figure(X, Y, Z, x, y, './figures/GDconvex2D.png')
    # plot_3D_figure(X, Y, Z, x, y, z, './figures/GDconvex3D.png')
    x_1, x_2, y_1, y_2, delta = -2.0, 2.0, -2.0, 2.0, 0.025
    x_start, y_start = 0.0, 0.0
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, rosenbrock)
    x, y, z = generate_points(x_start, y_start, rosenbrock, rosenbrock_gradient, 0.001, 10000)
    plot_2D_figure(X, Y, Z, x, y, './figures/GDrosenbrock2D.png')
    plot_3D_figure(X, Y, Z, x, y, z, './figures/GDrosenbrock3D.png')