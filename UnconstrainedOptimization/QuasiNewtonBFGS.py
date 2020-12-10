"""
演示拟牛顿法DFP解决一个凸函数的最优化问题的例子
优化函数：
        f(x,y)=x**2+3*y**2-2x*y-6
以及牛顿法解决一个非凸函数优化的例子：
优化函数：（著名的rosenbrock函数）
        f(x,y)=(1-x)**2 + 100 * (y-x**2)**2
"""
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cvx_function(X):
    """
    凸函数函数值的计算
    """
    z1 = X[0]**2
    z2 = X[1]**2
    z = z1 + 3 * z2 - 2 * X[0] * X[1] - 6
    return z


def cvx_fucntion_gradient(x):
    """
    计算凸函数的梯度
    """
    grad_x = 2 * x[0] - 2 * x[1]
    grad_y = 6 * x[1] - 2 * x[0]
    return np.array([grad_x, grad_y])


def rosenbrock(X):
    """
    非凸函数rosenbrock函数值计算
    """
    z1 = (1.0 - X[0])**2
    z2 = 100 * (X[1] - X[0]*X[0])**2
    z = z1 + z2
    return z


def rosenbrock_gradient(x):
    """
    计算非凸函数rosenbrock的梯度
    """
    grad_x = 400 * x[0] * x[0] * x[0] + 2 * x[0] - 400 * x[0] * x[1] - 2
    grad_y = 200 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])


def generate_grid(x_1, x_2, y_1, y_2, delta, f):
    """
    生成二维网格，并计算网格中各个点的值，用于后续画登高线三维图
    """
    x = np.arange(x_1, x_2, delta)
    y = np.arange(y_1, y_2, delta)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    return X, Y, Z


def plot_2D_figure(X, Y, Z, x, y, filepath):
    """
    画二维图
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
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x, y, z, c='r', linewidth=2)
    plt.colorbar(p, shrink=0.8)
    plt.savefig(filepath)
    plt.show()


def generate_points(x_start, f, grad, epsilon=1e-5, steps=100000):
    """
    根据拟牛顿法（DFP）生成优化点列的过程
    :param x_start: 起始点的坐标
    :param f: 需要优化的函数
    :param grad: 计算f函数的梯度函数
    :param epsilon: 迭代停止的条件，当当前点的梯度的模小于epsilon时，迭代停止
    :param steps: 最大的迭代步数
    :return: 优化过程生成点列的x坐标序列，y坐标序列，以及每一个点对应的函数值
    """
    X, x_old = x_start, x_start
    Z = f(x_start)
    H_old = np.mat(np.eye(2, dtype=np.float32) * 0.01)
    grad_old = np.mat(grad(x_old))
    I = np.mat(np.eye(2, dtype=np.float32))
    for i in range(1, steps):
        x_new = x_old - np.array(H_old * grad_old)
        grad_new = np.mat(grad(np.array(x_new)))
        if np.sqrt(np.sum(grad_new.T * grad_new)) < epsilon:
            X = np.concatenate((X, x_new), axis=1)
            z_new = f(np.array(x_new))
            Z = np.concatenate((Z, z_new))
            print("Convergence at step: ", i)
            print("Final varaible values: ", [x_new[0], x_new[1]])
            print("Final f(x,y):", z_new)
            break
        y = np.mat(grad_new - grad_old)
        s = np.mat(x_new - x_old)
        H_new = (I - (s * y.T) / (s.T * y)) * H_old * (I - (s * y.T) / (s.T * y)) + (s * s.T) / (s.T * y)
        X = np.concatenate((X, x_new), axis=1)
        z_new = f(np.array(x_new))
        Z = np.concatenate((Z, z_new))
        H_old = H_new
        grad_old = grad_new
        x_old = x_new
    return X[0], X[1], Z


if __name__ == "__main__":
    x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    x_start = np.array([[3.4], [3.5]])
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    x, y, z = generate_points(x_start, cvx_function, cvx_fucntion_gradient)
    plot_2D_figure(X, Y, Z, x, y, './figures/BFGSconvex2D.png')
    plot_3D_figure(X, Y, Z, x, y, z, './figures/BFGSconvex3D.png')
    # x_1, x_2, y_1, y_2, delta = -2.0, 2.0, -2.0, 2.0, 0.025
    # x_start = np.array([[0.0], [0.0]])
    # X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, rosenbrock)
    # x, y, z = generate_points(x_start, rosenbrock, rosenbrock_gradient)
    # plot_2D_figure(X, Y, Z, x, y, './figures/BFGSrosenbrock2D.png')
    # plot_3D_figure(X, Y, Z, x, y, z, './figures/BFGSrosenbrock3D.png')