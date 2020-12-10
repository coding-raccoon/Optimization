"""
演示坐标下降法解决一个凸函数的最优化问题的例子
优化函数：
        f(x,y)=x**2+3*y**2-2*x*y-6
"""
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def function(X, Y):
    """
    函数值的计算
    """
    z1 = X**2
    z2 = Y**2
    z = z1 + 3 * z2 - 2 * X * Y - 6
    return z


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


def plot_2D_figure(X, Y, Z, x, y):
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
    plt.savefig('./figures/cordinatedescent2D.png')
    plt.show()


def plot_3D_figure(X, Y, Z, x, y, z):
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
    plt.savefig('./figures/cordinatedescent3D.png')
    plt.show()
    pass


def generate_points(x_start, y_start, f=function, steps=200):
    """
    根据坐标下降法，生成优化过程中的点列
    :param x_start: 起始点x坐标
    :param y_start: 起始点y坐标
    :param f: 函数
    :param steps: 迭代的步数
    :return: 优化点列中各点x，y坐标以及对应的函数值
    """
    X, Y, Z = [x_start], [y_start], [f(x_start, y_start)]
    for i in range(1, steps):
        if i % 2 == 0:
            X.append(Y[i-1])
            Y.append(Y[i-1])
            Z.append(f(X[i], Y[i]))
        else:
            Y.append(X[i-1]/3)
            X.append(X[i-1])
            Z.append(f(X[i], Y[i]))
    return X, Y, Z


if __name__ == "__main__":
    x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    x_start, y_start = 3.4, 3.5
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, function)
    x, y, z = generate_points(x_start, y_start)
    plot_2D_figure(X, Y, Z, x, y)
    plot_3D_figure(X, Y, Z, x, y, z)