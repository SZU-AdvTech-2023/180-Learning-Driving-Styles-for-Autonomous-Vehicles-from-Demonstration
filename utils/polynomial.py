import numpy as np

# 四次多项式
class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

# 五次多项式
class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt

import torch
# 五次多项式
def cal_quintic_spline_point(a:torch.tensor, s, v):
    a0 = s
    a1 = v
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5)
    return torch.stack(res, dim=1)[:, 1:]

def cal_quintic_spline_first_derivative(a:torch.tensor, v):
    a1 = v
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4)
    return torch.stack(res, dim=1)[:,1:]

def cal_quintic_spline_second_derivative(a:torch.tensor):
    res = []
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    for i in range(51):
        t = i*0.1
        res.append(2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3)
    return torch.stack(res, dim=1)[:,1:]

def cal_quintic_spline_third_derivative(a:torch.tensor):
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(6 * a3 + 24 * a4 * t + 60 * a5 * t ** 2)
    return torch.stack(res, dim=1)[:,1:]

# 多段五次多项式
def cal_quintic_multiple_spline_point(a:torch.tensor, s, v):
    # 第一段轨迹
    a0 = s
    a1 = v
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(1, 11):
        t = i*0.1
        res.append(a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5)
    # 拼接轨迹
    for traj_i in range(4):
        a0 = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
        a1 = a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
        a2 = 2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
        a3 = a[:,4+traj_i*3]
        a4 = a[:,5+traj_i*3]
        a5 = a[:,6+traj_i*3]
        for i in range(1, 11):
            t = i*0.1
            res.append(a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5)
    return torch.stack(res, dim=1)

def cal_quintic_multiple_spline_first_derivative(a:torch.tensor, v):
    # 第一段轨迹
    a1 = v
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(1, 11):
        t = i*0.1
        res.append(a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4)
    # 拼接轨迹
    for traj_i in range(4):
        a1 = a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
        a2 = 2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
        a3 = a[:,4+traj_i*3]
        a4 = a[:,5+traj_i*3]
        a5 = a[:,6+traj_i*3]
        for i in range(1, 11):
            t = i*0.1
            res.append(a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4)
    return torch.stack(res, dim=1)

def cal_quintic_multiple_spline_second_derivative(a:torch.tensor):
    # 第一段轨迹
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(1, 11):
        t = i*0.1
        res.append(2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3)
    # 拼接轨迹
    for traj_i in range(4):
        a2 = 2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
        a3 = a[:,4+traj_i*3]
        a4 = a[:,5+traj_i*3]
        a5 = a[:,6+traj_i*3]
        for i in range(1, 11):
            t = i*0.1
            res.append(2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3)
    return torch.stack(res, dim=1)

def cal_quintic_multiple_spline_third_derivative(a:torch.tensor):
    # 第一段轨迹
    a3 = a[:,1]
    a4 = a[:,2]
    a5 = a[:,3]
    res = []
    for i in range(1, 11):
        t = i*0.1
        res.append(6 * a3 + 24 * a4 * t + 60 * a5 * t ** 2)
    # 拼接轨迹
    for traj_i in range(4):
        a3 = a[:,4+traj_i*3]
        a4 = a[:,5+traj_i*3]
        a5 = a[:,6+traj_i*3]
        for i in range(1, 11):
            t = i*0.1
            res.append(6 * a3 + 24 * a4 * t + 60 * a5 * t ** 2)
    return torch.stack(res, dim=1)

# 四次多项式
def cal_quartic_spline_point(a:torch.tensor,s,v):
    a0 = s
    a1 = v
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4)
    return torch.stack(res, dim=1)[:,1:]

def cal_quartic_spline_first_derivative(a:torch.tensor,v):
    a1 = v
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(a1 + 2 * a2 * t +  3 * a3 * t ** 2 + 4 * a4 * t ** 3)
    return torch.stack(res, dim=1)[:,1:]

def cal_quartic_spline_second_derivative(a:torch.tensor):
    a2 = a[:,0]
    a3 = a[:,1]
    a4 = a[:,2]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2)
    return torch.stack(res, dim=1)[:,1:]

def cal_quartic_spline_third_derivative(a:torch.tensor):
    a3 = a[:,1]
    a4 = a[:,2]
    res = []
    for i in range(51):
        t = i*0.1
        res.append(6 * a3 + 24 * a4 * t)
    return torch.stack(res, dim=1)[:,1:]
