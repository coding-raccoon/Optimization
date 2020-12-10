"""
演示牛顿法解决一个凸函数的最优化问题的例子
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


def cvx_hessian_inverse(x):
    """
    计算凸函数的hessian矩阵的逆矩阵，牛顿法的更新方向需要其与梯度的乘积
    """
    hessian_matrix = np.mat([[2, -2], [-2, 6]])
    hessian_inverse = np.linalg.pinv(hessian_matrix)
    return np.array(hessian_inverse)


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


def rosenbrock_hessain_inverse(x):
    grad_xx = 1200 * x[0] * x[0] + 2 - 400 * x[1]
    grad_xy = - 400 * x[1]
    grad_yx = - 400 * x[1]
    grad_yy = 200
    hessian_matrix = np.mat([[grad_xx, grad_xy], [grad_yx, grad_yy]])
    hessian_inverse = np.linalg.pinv(hessian_matrix)
    return np.array(hessian_inverse)


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


def generate_points(x_start, f, grad, hessian_inverse, epsilon=1e-10, steps=200):
    """
    根据牛顿法生成优化点列的过程
    :param x_start: 起始点的坐标
    :param f: 需要优化的函数
    :param grad: 计算f函数的梯度函数
    :param hessian_inverse: 计算f函数hessian矩阵的逆矩阵的函数
    :param epsilon: 迭代停止的条件，当当前点的梯度的模小于epsilon时，迭代停止
    :param steps: 最大的迭代步数
    :return: 优化过程生成点列的x坐标序列，y坐标序列，以及每一个点对应的函数值
    """
    X = x_start
    Z = f(x_start)
    print(Z)
    for i in range(1, steps):
        current_grad = grad(X[:, i-1])
        if np.sqrt(np.sum(current_grad**2)) < epsilon:
            print("Convergence at step: ", i)
            break
        current_hessain_inverse = hessian_inverse(X[:, i-1])
        x_new = X[:, i-1].reshape(2, 1) - np.dot(current_hessain_inverse, current_grad)
        z_new = f(x_new)
        print(z_new)
        X = np.concatenate((X, x_new), axis=1)
        Z = np.concatenate((Z, z_new))
    return X[0], X[1], Z


if __name__ == "__main__":
    # x_1, x_2, y_1, y_2, delta = -4.0, 4.0, -4.0, 4.0, 0.025
    # x_start= np.array([[3.4], [3.5]])
    # X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, cvx_function)
    # x, y, z = generate_points(x_start, cvx_function, cvx_fucntion_gradient, cvx_hessian_inverse)
    # plot_2D_figure(X, Y, Z, x, y, './figures/Newtonconvex2D.png')
    # plot_3D_figure(X, Y, Z, x, y, z, './figures/Newtonconvex3D.png')
    x_1, x_2, y_1, y_2, delta = -2.0, 2.0, -2.0, 2.0, 0.025
    x_start = np.array([[0.0], [0.0]])
    X, Y, Z = generate_grid(x_1, x_2, y_1, y_2, delta, rosenbrock)
    x, y, z = generate_points(x_start, rosenbrock, rosenbrock_gradient, rosenbrock_hessain_inverse)
    plot_2D_figure(X, Y, Z, x, y, './figures/Newtonrosenbrock2D.png')
    plot_3D_figure(X, Y, Z, x, y, z, './figures/Newtonrosenbrock3D.png')