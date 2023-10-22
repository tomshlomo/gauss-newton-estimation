from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.linalg

Array = npt.NDArray[np.float64]

class Sensor(Protocol):
    def forward(self, x: Array) -> Array:
        pass

    def grad(self, x: Array) -> Array:
        pass

    def cov(self, x: Array) -> Array:
        pass


@dataclass
class Range(Sensor):
    loc: Array
    sigma: float

    """
    range = ||x - loc||
    """
    def forward(self, x: Array) -> Array:
        z = np.linalg.norm(x - self.loc)
        return z.reshape(1)

    def grad(self, x: Array) -> Array:
        r = x - self.loc
        g = r / np.linalg.norm(r)
        return g.reshape((1, -1))

    def cov(self, x: Array) -> Array:
        return np.reshape(self.sigma ** 2, (1,1))


def simulate():
    rng = np.random.default_rng(42)
    n_sensors = 10
    locs = rng.random((n_sensors, 2)) * 10 + 10
    # x_true = rng.random(2)
    x_true = np.zeros(2)
    measurements = []
    for loc in locs:
        sensor = Range(loc, 0.001)
        y = sensor.forward(x_true)
        cov = sensor.cov(x_true)
        y += rng.multivariate_normal(np.zeros_like(y), cov)
        measurements.append((sensor, y))
    x = fuse(measurements, np.zeros_like(x_true), x_true)
    err = x - x_true
    import plotly.express as px
    import plotly.io
    plotly.io.renderers.default = "browser"
    px.line(err).show()
    pass


def fuse(
    measurements: list[tuple[Sensor, Array]],
    x0: Array,
    x_true: Array,
) -> tuple[Array, Array]:
    """
    Linear, bayesian case:
    y = Ax + w
    w ~= N(0, C)
    x ~ N(m, P)

    E[x|y] = m + cov[x, y] cov[y]^-1 (y - E[y])
    cov[x, y] = cov[x, Ax + w]
              = cov[x] A'
              = P A'
    cov[y] = cov[Ax + w]
           = cov[Ax] + cov[w]
           = APA' + C
    E[y] = Am
    =>
    E[x|y] = m + PA'(APA'+C)^-1 (y - Am)


    Nonlinear, non bayesian:
    y ~= N(f(x), C(x))
    x_hat = argmin || f(x) - y ||^2 _C(x)
         ~= argmin || f(x_k) + grad f(x_k) (x - x_k) - y ||^2 _C(x_k)
    d (|| f(x_k) + grad f(x_k) (x - x_k) - y ||^2 _C(x_k)) / d(x)
    = grad f(x_k)' C(x_k)^-1 (f(x_k) + grad f(x_k) (x - x_k) - y) = 0
    =>
    solve for x:
    grad f(x_k)' C(x_k)^-1 grad f(x_k) (x - x_k) = = grad f(x_k)' C(x_k)^-1(y - f(x_k))
    """
    x = [x0]
    for k in range(100):
        x_prev = x[-1]
        y = []
        grad = []
        c = []
        f = []
        for sensor, y_i in measurements:
            y.append(y_i)
            c.append(np.linalg.inv(sensor.cov(x_prev)))
            grad.append(sensor.grad(x_prev))
            f.append(sensor.forward(x_prev))
        y = np.concatenate(y)
        grad = np.concatenate(grad)
        c = scipy.linalg.block_diag(*c)
        f = np.concatenate(f)

        err = y - f
        h = grad.T @ c
        a = h @ grad
        b = h @ err
        step = np.linalg.solve(a, b)
        x_new = x_prev + step
        print(np.linalg.norm(x_true - x_new))
        x.append(x_new)
    return np.array(x)

if __name__ == '__main__':
    simulate()