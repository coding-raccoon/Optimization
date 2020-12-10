"""
演示拟共轭梯度法解决一个凸函数的最优化问题的例子
优化函数：
        f(x,y)=x**2+3*y**2-2x*y-6
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


def generate_points(x_start, f, grad, epsilon=1e-5, steps=10):
    X, x_old = x_start, x_start
    Z = f(x_start)
    Q = np.mat([[2, -2], [-2, 6]])
    grad_old = np.mat(grad(x_old))
    d = - grad_old           # 初始化共轭方向为初始梯度方向
    for i in range(1, steps):
        if np.sqrt(np.sum(grad_old.T * grad_old)) < epsilon:      # 判断是否收敛到极小值点
            X = np.concatenate((X, x_old), axis=1)
            z_new = f(np.array(x_old))
            Z = np.concatenate((Z, z_new))
            print("Convergence at step: ", i)
            print("Final varaible values: ", [x_old[0], x_old[1]])
            print("Final f(x,y):", z_new)
            break
        alpha = (grad_old.T * grad_old) / (d.T * Q * d)     # 计算步长
        x_new = x_old + np.array(alpha)[0][0] * np.array(d)         # 计算下一步的点
        grad_new = np.mat(grad(x_new))      # 计算新的点的梯度
        beta = (grad_new.T * grad_new) / (grad_old.T * grad_old)        # 计算beta
        d = -1.0 * grad_new + np.array(beta)[0][0] * d      # 根据梯度和上一步共轭方向构造新的共轭方向
        X = np.concatenate((X, x_new), axis=1)
        z_new = f(np.array(x_new))
        Z = np.concatenate((Z, z_new))
        grad_old = grad_new
        x_old = x_new
    return X[0], X[1], Z


if __name__ == "__main__":
    x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    x_start = np.array([[3.4], [3.5]])
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    x, y, z = generate_points(x_start, cvx_function, cvx_fucntion_gradient)
    plot_2D_figure(X, Y, Z, x, y, './figures/Conjugate2D.png')
    plot_3D_figure(X, Y, Z, x, y, z, './figures/Conjugate3D.png')
