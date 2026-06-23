import numpy as np
from scipy.linalg import solve, cho_factor, cho_solve
from scipy.stats import chi2


def mahalanobis_distance_kalman(residual, innovation_cov):
    residual = np.asarray(residual).reshape(-1, 1)  

    try:
        
        c, low = cho_factor(innovation_cov)
        sol = cho_solve((c, low), residual)  
        dist_sq = float(residual.T @ sol)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(innovation_cov)
        dist_sq = float(residual.T @ inv_cov @ residual)

    return dist_sq


class KalmanFilter:
    def __init__(self):

        dt = 1.0  
        self.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        self.Q = np.eye(8) * 0.1

        self.R = np.eye(4) * 10

        self.P = np.eye(8) * 1000.0

        self.x = np.zeros((8, 1))

    def _predict(self):
        return np.dot(self.F, self.x)

    def predict(self):

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.x[2][0] = min(self.old_x[2] * 2, max(self.old_x[2] * 0.5, self.x[2][0]))
        self.x[3][0] = min(self.old_x[2] * 2, max(self.old_x[3] * 0.5, self.x[3][0]))
        return self.x

    def mahalanobis_distance_kalman(self, z):
        residual = z.reshape(-1, 1) - np.dot(self.H, self.x)
        S = self.H @ self.P @ self.H.T  
        return mahalanobis_distance_kalman(residual, S)

    def is_in_mahalanobis(self, z, confidence=0.95):
        residual = z.reshape(-1, 1) - np.dot(self.H, self.x)
        S = self.H @ self.P @ self.H.T + self.R
        d = residual.shape[0]
        mahal_sq = mahalanobis_distance_kalman(residual, S)
        threshold = chi2.ppf(confidence, df=d)
        return mahal_sq < threshold

    def update(self, z, ocur=False):
        self.old_x = z
        for u, val in zip(range(4), [z[2], z[3], z[2], z[3]]):
            self.R[u, u] = val ** 0.5
            self.Q[u, u] = (val) ** 0.5
            self.Q[u + 4, u] = (val) ** 0.5

        y = z.reshape(-1, 1) - np.dot(self.H, self.x)

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        if ocur:
            self.x = self.x + np.dot(K, y) * ([[1]] * 4 + [[0]] * 4)
        else:
            self.x = self.x + np.dot(K, y)

        I = np.eye(self.F.shape[0])
        self.P = (np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) +
                  np.dot(np.dot(K, self.R), K.T))

        return self.x[:, 0].tolist()

    def init_state(self, z):

        
        self.x = np.array([z[0], z[1], z[2], z[3], 0, 0, 0, 0]).reshape(-1, 1)
        self.P = np.eye(8) * 1000.0  
        self.old_x = z


class RLS:
    def __init__(self, lam=0.99, delta=50):
        
        self.n_inputs = 2 * 11  
        self.n_outputs = 2  
        self.lam = np.clip(lam, 0.9, 1.0)  

        
        self.aug_dim = self.n_inputs + 1
        
        self.P = delta * np.eye(self.aug_dim, dtype=np.float64)
        
        
        
        
        self.Theta = np.zeros((self.n_outputs, self.aug_dim), dtype=np.float64)
        
        self._K = np.zeros((self.aug_dim, 1), dtype=np.float64)

    def _process_input_11x2(self, X):
        
        X_arr = np.asarray(X, dtype=np.float64)

        
        if X_arr.ndim != 2 or X_arr.shape[1] != 2:
            raise ValueError(f"输入第二维错误！要求 2 列，当前输入形状为 {X_arr.shape}")

        
        n_rows = X_arr.shape[0]
        target_rows = 11

        if n_rows < target_rows:
            
            first_row = X_arr[0:1, :]  
            
            pad_rows = target_rows - n_rows
            
            pad_data = np.repeat(first_row, pad_rows, axis=0)
            
            X_padded = np.concatenate([pad_data, X_arr], axis=0)
        elif n_rows > target_rows:
            
            X_padded = X_arr[:target_rows, :]
        else:
            
            X_padded = X_arr
        
        
        x_flat = X_padded.flatten()
        phi = np.concatenate([x_flat, np.array([1.0], dtype=np.float64)])

        
        return phi.reshape(-1, 1)

    def update(self, X_11x2, y_2d):
        
        phi = self._process_input_11x2(X_11x2)

        
        denominator = self.lam + phi.T @ self.P @ phi
        K = (self.P @ phi) / denominator

        y_pred = self.Theta @ phi
        
        e = y_2d.reshape(-1, 1) - y_pred
        
        e_std = np.std(e) if np.std(e) > 1e-6 else 1.0
        e = np.clip(e, -10 * e_std, 10 * e_std)
        
        self.Theta = self.Theta + e @ K.T

        
        

        return self.Theta

    def predict(self, X_11x2):
        
        phi = self._process_input_11x2(X_11x2)

        
        y_pred = []
        for i in range(self.n_outputs):
            theta_i = self.Theta[i, :].reshape(-1, 1)
            y_i_pred = float(theta_i.T @ phi)
            y_pred.append(y_i_pred)

        return y_pred


class ScalarKalmanRegressor:


    def __init__(self, Q_diag=(1e-4, 1), R=1e6, P0_scale=1e8, LAM=0.999):
        self.input_dim = 1
        Q_diag = [Q_diag[0]] * self.input_dim + [Q_diag[1]]

        self.theta = np.zeros(self.input_dim + 1, dtype=np.float64)  
        self.P = np.eye(self.input_dim + 1, dtype=np.float64) * P0_scale
        self.Q = np.diag(Q_diag).astype(np.float64)
        self.R = float(R)
        self.LAM = LAM
        self.cnt = 0

    def process(self, x):
        if len(x) < self.input_dim:
            pad_data = np.repeat(x[0], self.input_dim - len(x), axis=0)
            x = np.concatenate([pad_data, x], axis=0)
        elif len(x) > self.input_dim:
            x = x[-self.input_dim:]
        x = np.concatenate([x, np.array([1])])
        return x

    def update(self, x, y):

        phi = self.process(x)

        
        theta_pred = self.theta.copy()

        if self.cnt > 10:
            P_pred = self.P / self.LAM + self.Q
        else:
            P_pred = self.P + self.Q
        self.cnt += 1

        
        y_pred_prior = phi @ theta_pred
        e = y - y_pred_prior
        S = phi @ P_pred @ phi + self.R
        K = (P_pred @ phi) / S

        
        self.theta = theta_pred + K * e
        self.P = (np.eye(self.input_dim + 1) - np.outer(K, phi)) @ P_pred
        self.P = 0.5 * (self.P + self.P.T)  

        
        y_pred_post = phi @ self.theta

        return {
            'theta': self.theta[0],
            'y_pred_prior': y_pred_prior,
            'y_pred_post': y_pred_post,
            'error_prior': e,
            'error_post': y - y_pred_post
        }

    def predict(self, x):
        

        x = self.process(x)
        return np.dot(x, self.theta)

    def get_params(self):
        
        return self.theta











































































