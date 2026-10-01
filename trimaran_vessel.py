# 三胴船の数理モデル Python版
# Author: 加地勇佑
# License: GPLv3
# 作成日: 2025-05-29

import numpy as np
from numpy import sin, cos, pi, exp, random

class trimaran_vessel():
    # パラメータ
    m = 1256  # 質量[g]
    m_x = 120  # 付加質量[g]
    m_y = 240  # 付加質量[g]
    I_z = 50.7

    K_Rx = 857
    K_Ry = 865
    K_Rr = 79.7

    K_Fu = 13.6
    K_Fv = 13.8
    K_T = 2.5

    epsilon = 1e-6

    def __init__(self):
        self.x = np.zeros((6))
        # 状態 [N, E, theta, u, v, r]
        
        self.u = np.zeros((6))
        # 制御入力 [u1, u2, u3, alpha1, alpha2, alpha3]

        self.first_step = True
        self.previous_u = np.zeros((6))
        # 前回の制御入力 [u1, u2, u3, alpha1, alpha2, alpha3]
        
        self.tau = -0.5

        self.rng = random.default_rng()
        self.noise_base = self.rng.integers(low=5, high=10)
        return
    
    def set_first_order_lag_tau(self, tau):
        # 1次遅れ系の時間定数 tau を設定する

        self.tau = tau
        return

    def first_order_lag(self, u_target, u_previous, dt):
        # 制御入力の反映に1次遅れ系を適用

        if self.tau < self.epsilon:
            u_lag = u_target
        else:
            u_lag = u_previous + (u_target - u_previous) * (1 - exp(-dt / self.tau))
        return u_lag

    def get_state(self):
        # 現在の状態 x[N, E, theta, u, v, r] + u[u1, u2, u3, alpha1, alpha2, alpha3] を取得する

        state = np.concatenate((self.x, self.u))
        return state

    def update_state(self, dt=0.1):
        # 状態 x[N, E, theta, u, v, r] を更新する

        if self.first_step:
            self.previous_u = self.u.copy()
            self.first_step = False
        self.u = self.first_order_lag(self.u, self.previous_u, dt)
        self.previous_u = self.u.copy()
        # 制御入力の1次遅れ系を適用
        
        x0 = self.x.copy()
        u0 = self.u.copy()
        k1 = self.fn(x0, u0)
        k2 = self.fn(x0 + 0.5 * dt * k1, u0)
        k3 = self.fn(x0 + 0.5 * dt * k2, u0)
        k4 = self.fn(x0 + dt * k3, u0)
        self.x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # 4次のルンゲクッタ法で状態を更新

        # dx = self.fn(self.x, self.u)
        # self.x += dx * dt
        # オイラー法で状態を更新

        self.x[2] = (self.x[2] + pi) % (2 * pi) - pi
        # 角度を -π から π の範囲に制限
        return
    
    def set_control_input(self, u):
        # 制御入力 u[u1, u2, u3, alpha1, alpha2, alpha3] を設定する

        self.u = np.array(u)
        return
    
    def noise(self):
        # 定数項+ホワイトノイズ
        return self.noise_base + self.rng.random()


    def fn(self, x, u):
        # 船の状態方程式　KT = 2.5

        X_1 = (
            -self.K_Rx * x[3] * abs(x[3])
            + self.K_T * u[0] * cos(u[3])
            + (
                -self.K_Fu * x[3] * abs(x[3]) * sin(u[3])
                + self.K_Fv * x[4] * abs(x[4]) * cos(u[3])
            )
            * sin(u[3])
        )
        X_2 = (
            -self.K_Rx * x[3] * abs(x[3])
            + self.K_T * u[1] * cos(u[4])
            + (
                -self.K_Fu * x[3] * abs(x[3]) * sin(u[4])
                + self.K_Fv * x[4] * abs(x[4]) * cos(u[4])
            )
            * sin(u[4])
        )
        X_3 = (
            -self.K_Rx * x[3] * abs(x[3])
            + self.K_T * u[2] * cos(u[5])
            + (
                -self.K_Fu * x[3] * abs(x[3]) * sin(u[5])
                + self.K_Fv * x[4] * abs(x[4]) * cos(u[5])
            )
            * sin(u[5])
        )
        Y_1 = (
            -self.K_Ry * x[4] * abs(x[4])
            + self.K_T * u[0] * sin(u[3])
            + (
                self.K_Fu * x[3] * abs(x[3]) * sin(u[3])
                - self.K_Fv * x[4] * abs(x[4]) * cos(u[3])
            )
            * cos(u[3])
        )
        Y_2 = (
            -self.K_Ry * x[4] * abs(x[4])
            + self.K_T * u[1] * sin(u[4])
            + (
                self.K_Fu * x[3] * abs(x[3]) * sin(u[4])
                - self.K_Fv * x[4] * abs(x[4]) * cos(u[4])
            )
            * cos(u[4])
        )
        Y_3 = (
            -self.K_Ry * x[4] * abs(x[4])
            + self.K_T * u[2] * sin(u[5])
            + (
                self.K_Fu * x[3] * abs(x[3]) * sin(u[5])
                - self.K_Fv * x[4] * abs(x[4]) * cos(u[5])
            )
            * cos(u[5])
        )
        X_H = X_1 + X_2 + X_3 + self.noise()
        Y_H = Y_1 + Y_2 + Y_3 + self.noise()
        N_H = 0.23 * (
            Y_1
            + X_2 * cos(pi / 6)
            - Y_2 * sin(pi / 6)
            - X_3 * cos(pi / 6)
            - Y_3 * sin(pi / 6)
        ) - self.K_Rr * x[5] * abs(x[5])

        dx = np.zeros((6))
        dx[0] = x[3] * cos(x[2]) - x[4] * sin(x[2])
        dx[1] = x[3] * sin(x[2]) + x[4] * cos(x[2])
        dx[2] = x[5]
        dx[3] = (self.m * x[4] * x[5] + X_H) / (self.m + self.m_x)
        dx[4] = (-self.m * x[3] * x[5] + Y_H) / (self.m + self.m_y)
        dx[5] = N_H / (self.I_z)
        return dx
