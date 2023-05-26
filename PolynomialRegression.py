import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


class PolyRegress(object):
    ###
    # borrowed covariance equations from
    # https://xavierbourretsicotte.github.io/stats_inference_2.html#Custom-Python-class
    # but did not use their weird definition for "R^2"
    # reference for R^2: https://en.wikipedia.org/wiki/Coefficient_of_determination
    ###
    def __init__( self, X, Y, P=1, fit=False, pass_through_origin=False, log=False, ln=False, dtype=np.float64, ):
        
        self.log_func = np.log10 if log else np.log
        self.exp_func = lambda x: np.power(10, x) if log else np.exp
        keep = np.isfinite(X + Y)
        if log or ln:
            keep = keep & (X > 0) & (Y > 0)
            X, Y = self.log_func(X[keep]), self.log_func(Y[keep])
        else:
            X, Y = X[keep], Y[keep]
        
        self.log = log
        self.ln = ln

        self.X = X.astype(dtype)
        self.Y = Y.astype(dtype)

        self.N = len(self.X)
        self.P = P + 1
        # A is the matrix for X
        self.A = np.array([self.X ** i for i in range(self.P)]).T
        if pass_through_origin:
            self.A[:, 0] = 0

        self.XX = np.dot(self.A.T, self.A)
        self.X_bar = np.mean(self.X, 0)
        self.y_bar = np.mean(self.Y)

        self.b, self.cov, self.err = None, None, None
        self.scatter = None
        self.norm_resid = None
        self.y_hat = None
        self.R2, self.R2_a = None, None

        if fit:
            self.fit()

    def __str__(self):
        if self.b is not None:
            b = self.b
            e = self.err
            s = self.scatter
            term = lambda i: f"({self.b[i]:0.3g}+/-{e[i]:0.3g}) * x^{i}"
            out1 = " + ".join([term(i) for i in range(self.P)])
            if (not self.log) or (not self.ln):
                # out=f'm:{b[1]:0.3g}+/-{e[1]:0.3g}, b:{b[0]:0.3g}+/-{e[0]:0.3g}, scatter:{s:0.3g}'
                out = out1 + f" , scatter:{s:0.3g}"
            elif self.log:
                # out=f'm:{b[1]:0.3g}+/-{e[1]:0.3g}, b:{b[0]:0.3g}+/-{e[0]:0.3g}, scatter:{s:0.3g}'
                out = out1 + f" , scatter:{s:0.3g}"
                out = f"{out}\n+10^b:{10**b[0]:0.3g}"
            elif self.ln:
                out = out1 + f" , scatter:{s:0.3g}"
                out = f"{out}\n+10^b:{np.exp(b[0]):0.3g}"
        else:
            out = "\nERROR::Has not been run. run model.fit() now\n"
        return "\n" + out + "\n"

    def __repr__(self):
        out1 = f"\norder={self.P-1} regression"
        out2 = self.__str__()
        return out1 + out2

    def fit(self):
        if np.linalg.det(self.XX) != 0:
            # self.b = np.dot(np.dot(np.linalg.inv(self.XX),self.A.T),self.Y)
            self.b = np.linalg.solve(np.dot(self.A.T, self.A), np.dot(self.A.T, self.Y))
        else:
            self.b, *_ = np.linalg.lstsq(self.A, self.Y, rcond=-1,)
            # self.b = np.dot(np.dot(np.linalg.pinv(self.XX),self.A.T),self.Y)

        self.y_hat = np.dot(self.A, self.b)

        # Sum of squares
        SS_res = np.sum((self.Y - self.y_hat) ** 2)
        SS_tot = np.sum((self.Y - self.y_bar) ** 2)
        SS_exp = np.sum((self.y_hat - self.y_bar) ** 2)
        R2 = 1 - SS_res / SS_tot

        self.residual = self.y_hat - self.Y

        # R squared and adjusted R-squared
        self.R2 = R2  # Use more general definition SS_exp / SS_tot
        self.R2_a = (self.R2 * (self.N - 1) - self.P) / (self.N - self.P - 1)

        # Variances and standard error of coefficients
        self.norm_resid = SS_res / (self.N - self.P - 1)
        self.cov = self.norm_resid * np.linalg.pinv(self.XX)
        self.err = np.sqrt(np.diag(self.cov))

        # ortho_dist = (self.Y - self.b[1] * self.X - self.b[0])/np.sqrt(1 + self.b[1]**2)
        # self.scatter = np.std(ortho_dist)/np.cos(np.arctan(self.b[1]))
        self.scatter = np.std(self.residual)

        if self.log:
            self.percent_scatter = np.mean(
                np.abs(10 ** self.y_hat - 10 ** self.Y) / 10 ** self.Y
            )
        elif self.ln:
            self.percent_scatter = np.mean(
                np.abs(np.exp(self.y_hat) - np.exp(self.Y)) / np.exp(self.Y)
            )
        else:
            self.percent_scatter = np.mean(self.residual / self.Y)

        return self.b, self.err

    def func(self, x):
        p = self.b[::-1]
        if self.log:
            return 10 ** np.polyval(p, self.log_func(x))
        elif self.ln:
            return np.exp(np.polyval(p, self.log_func(x)))
        else:
            return np.polyval(p, x)

    def sample_covarariance(self, n=10000):
        return np.random.multivariate_normal(self.b, self.cov, n)

    def plot(
        self,
        ax=None,
        color="",
        data=False,
        marker="o",
        ms=5,
        mec="none",
        unlog=False,
        unln = False,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()
            
        if self.log or unlog or self.ln or unln:
            x = self.log_func(self.X)
            y = self.log_func(self.Y)
        else:
            x = self.X
            y = self.Y
        if self.log or unlog:
            xx = np.logspace(self.X.min(), self.X.max(), 30)
        elif self.ln or unln:
            xx = np.logspace(self.X.min(), self.X.max(), 30,base=np.e)
        else:
            xx = np.linspace(self.X.min(), self.X.max(), 30)

        if color == "":
            c = ("k", "r")

        if isinstance(color, tuple):
            c_data = color[0]
            c_line = color[1]
            data = True
        elif color == "":
            if data:
                c_data = plt.plot([], [], "-")[0].get_color()
                c_line = "k"
            else:
                c_line = "k"
        else:
            c_line = color
            if data:
                c_data = plt.plot([], [], "-")[0].get_color()

        if data:
            p = ax.plot(x, y, color=c_data, ms=ms, mec=mec, marker=marker, ls="")

        ax.plot(xx, self.func(xx), "-", color=c_line, **kwargs)

        if self.log or self.ln:
            ax.set_xscale("log")
            ax.set_yscale("log")

