"""
Stochastic Differential Equation (SDE) Solvers

Implements numerical methods for SDEs:
dx_t = f(x_t, t)dt + g(x_t, t)dW_t

where W_t is a Wiener process (Brownian motion).

Methods:
- Euler-Maruyama (order 0.5 strong convergence)
- Milstein (order 1.0 strong convergence)
- Runge-Kutta for SDEs
- Jump-diffusion processes (Merton, Kou)

Applications:
- Continuous-time geopolitical dynamics
- Financial contagion models
- Regime transitions with stochastic shocks
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from scipy.stats import poisson, norm


@dataclass
class SDESolution:
    """
    Solution to an SDE.

    Attributes
    ----------
    t : np.ndarray
        Time points
    x : np.ndarray
        State trajectories
    method : str
        Integration method used
    """
    t: np.ndarray
    x: np.ndarray
    method: str


class SDESolver:
    """
    Base class for SDE solvers.

    Solves: dx_t = f(x_t, t)dt + g(x_t, t)dW_t
    """

    def __init__(
        self,
        drift: Callable[[np.ndarray, float], np.ndarray],
        diffusion: Callable[[np.ndarray, float], np.ndarray],
        x0: np.ndarray,
        t0: float = 0.0
    ):
        """
        Initialize SDE solver.

        Parameters
        ----------
        drift : callable
            Drift function f(x, t)
        diffusion : callable
            Diffusion function g(x, t)
        x0 : np.ndarray
            Initial condition
        t0 : float
            Initial time
        """
        self.drift = drift
        self.diffusion = diffusion
        self.x0 = np.asarray(x0)
        self.t0 = t0
        self.dim = len(self.x0)

    def integrate(
        self,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> SDESolution:
        """
        Integrate SDE.

        Parameters
        ----------
        T : float
            Final time
        dt : float
            Time step
        n_paths : int
            Number of sample paths

        Returns
        -------
        SDESolution
            Solution object
        """
        raise NotImplementedError("Subclasses must implement integrate()")


class EulerMaruyama(SDESolver):
    """
    Euler-Maruyama method for SDEs.

    Simplest method with order 0.5 strong convergence.

    x_{n+1} = x_n + f(x_n, t_n)Δt + g(x_n, t_n)ΔW_n

    where ΔW_n ~ N(0, Δt)
    """

    def integrate(
        self,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> SDESolution:
        """
        Integrate using Euler-Maruyama.

        Parameters
        ----------
        T : float
            Final time
        dt : float
            Time step
        n_paths : int
            Number of paths to simulate

        Returns
        -------
        SDESolution
            Solution
        """
        # Time grid
        n_steps = int((T - self.t0) / dt)
        t = np.linspace(self.t0, T, n_steps + 1)

        # Initialize paths
        x = np.zeros((n_paths, n_steps + 1, self.dim))
        x[:, 0, :] = self.x0

        # Brownian increments
        sqrt_dt = np.sqrt(dt)

        # Simulate paths
        for i in range(n_steps):
            t_current = t[i]

            for path in range(n_paths):
                x_current = x[path, i, :]

                # Drift term
                drift_term = self.drift(x_current, t_current) * dt

                # Diffusion term
                dW = np.random.randn(self.dim) * sqrt_dt
                diffusion_term = self.diffusion(x_current, t_current) * dW

                # Update
                x[path, i + 1, :] = x_current + drift_term + diffusion_term

        return SDESolution(t=t, x=x, method='euler_maruyama')


class Milstein(SDESolver):
    """
    Milstein method for SDEs.

    Higher-order method with order 1.0 strong convergence.
    Requires derivative of diffusion term.

    x_{n+1} = x_n + f(x_n)Δt + g(x_n)ΔW_n
              + 0.5 * g(x_n) * g'(x_n) * ((ΔW_n)^2 - Δt)

    where g'(x) = ∂g/∂x
    """

    def __init__(
        self,
        drift: Callable,
        diffusion: Callable,
        diffusion_derivative: Callable,
        x0: np.ndarray,
        t0: float = 0.0
    ):
        """
        Initialize Milstein solver.

        Parameters
        ----------
        drift : callable
            Drift function
        diffusion : callable
            Diffusion function
        diffusion_derivative : callable
            Derivative of diffusion: ∂g/∂x
        x0 : np.ndarray
            Initial condition
        t0 : float
            Initial time
        """
        super().__init__(drift, diffusion, x0, t0)
        self.diffusion_derivative = diffusion_derivative

    def integrate(
        self,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> SDESolution:
        """
        Integrate using Milstein method.

        Parameters
        ----------
        T : float
            Final time
        dt : float
            Time step
        n_paths : int
            Number of paths

        Returns
        -------
        SDESolution
            Solution
        """
        n_steps = int((T - self.t0) / dt)
        t = np.linspace(self.t0, T, n_steps + 1)

        x = np.zeros((n_paths, n_steps + 1, self.dim))
        x[:, 0, :] = self.x0

        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            t_current = t[i]

            for path in range(n_paths):
                x_current = x[path, i, :]

                # Drift
                drift_term = self.drift(x_current, t_current) * dt

                # Diffusion
                dW = np.random.randn(self.dim) * sqrt_dt
                g = self.diffusion(x_current, t_current)
                diffusion_term = g * dW

                # Milstein correction term
                g_prime = self.diffusion_derivative(x_current, t_current)
                correction = 0.5 * g * g_prime * ((dW**2) - dt)

                # Update
                x[path, i + 1, :] = x_current + drift_term + diffusion_term + correction

        return SDESolution(t=t, x=x, method='milstein')


class StochasticRungeKutta(SDESolver):
    """
    Stochastic Runge-Kutta method.

    Higher-order method for SDEs with better accuracy.
    """

    def integrate(
        self,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> SDESolution:
        """
        Integrate using stochastic Runge-Kutta.

        Parameters
        ----------
        T : float
            Final time
        dt : float
            Time step
        n_paths : int
            Number of paths

        Returns
        -------
        SDESolution
            Solution
        """
        n_steps = int((T - self.t0) / dt)
        t = np.linspace(self.t0, T, n_steps + 1)

        x = np.zeros((n_paths, n_steps + 1, self.dim))
        x[:, 0, :] = self.x0

        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            t_current = t[i]

            for path in range(n_paths):
                x_current = x[path, i, :]

                # Generate Wiener increments
                dW = np.random.randn(self.dim) * sqrt_dt

                # Stage 1
                k1_drift = self.drift(x_current, t_current)
                k1_diff = self.diffusion(x_current, t_current)

                # Stage 2 (predictor)
                x_pred = x_current + k1_drift * dt + k1_diff * dW

                k2_drift = self.drift(x_pred, t_current + dt)
                k2_diff = self.diffusion(x_pred, t_current + dt)

                # Update (corrector)
                drift_term = 0.5 * (k1_drift + k2_drift) * dt
                diffusion_term = 0.5 * (k1_diff + k2_diff) * dW

                x[path, i + 1, :] = x_current + drift_term + diffusion_term

        return SDESolution(t=t, x=x, method='stochastic_rk')


class JumpDiffusionProcess:
    """
    Jump-diffusion process (Merton model).

    Combines continuous diffusion with discrete jumps:
    dx_t = μ x_t dt + σ x_t dW_t + x_t dJ_t

    where J_t is a compound Poisson process:
    - Jumps occur with intensity λ
    - Jump sizes Y ~ N(μ_J, σ_J^2)
    """

    def __init__(
        self,
        drift: float,
        diffusion: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
        x0: np.ndarray
    ):
        """
        Initialize jump-diffusion process.

        Parameters
        ----------
        drift : float
            Drift coefficient μ
        diffusion : float
            Diffusion coefficient σ
        jump_intensity : float
            Jump intensity λ (expected number of jumps per unit time)
        jump_mean : float
            Mean jump size (log-normal)
        jump_std : float
            Jump size standard deviation
        x0 : np.ndarray
            Initial condition
        """
        self.drift = drift
        self.diffusion = diffusion
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.x0 = np.asarray(x0)
        self.dim = len(self.x0)

    def simulate(
        self,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> SDESolution:
        """
        Simulate jump-diffusion paths.

        Parameters
        ----------
        T : float
            Final time
        dt : float
            Time step
        n_paths : int
            Number of paths

        Returns
        -------
        SDESolution
            Solution
        """
        n_steps = int(T / dt)
        t = np.linspace(0, T, n_steps + 1)

        x = np.zeros((n_paths, n_steps + 1, self.dim))
        x[:, 0, :] = self.x0

        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            for path in range(n_paths):
                x_current = x[path, i, :]

                # Continuous part (Euler-Maruyama)
                dW = np.random.randn(self.dim) * sqrt_dt
                continuous = self.drift * x_current * dt + self.diffusion * x_current * dW

                # Jump part (Poisson process)
                n_jumps = poisson.rvs(self.jump_intensity * dt)

                jump_total = 0.0
                if n_jumps > 0:
                    # Sample jump sizes
                    jump_sizes = norm.rvs(
                        loc=self.jump_mean,
                        scale=self.jump_std,
                        size=n_jumps
                    )
                    # Convert to multiplicative jumps
                    jump_total = x_current * np.sum(np.exp(jump_sizes) - 1)

                # Update
                x[path, i + 1, :] = x_current + continuous + jump_total

                # Ensure non-negative (if applicable)
                x[path, i + 1, :] = np.maximum(x[path, i + 1, :], 0)

        return SDESolution(t=t, x=x, method='jump_diffusion')


class GeopoliticalSDE:
    """
    Geopolitical system as continuous-time SDE.

    Models geopolitical variables as SDEs with:
    - Continuous dynamics (drift + diffusion)
    - Discrete shocks (jumps)
    - Regime-dependent parameters
    """

    def __init__(
        self,
        variable_names: List[str],
        drift_functions: Dict[str, Callable],
        diffusion_functions: Dict[str, Callable],
        jump_intensities: Optional[Dict[str, float]] = None
    ):
        """
        Initialize geopolitical SDE.

        Parameters
        ----------
        variable_names : list
            Names of state variables
        drift_functions : dict
            Drift function for each variable
        diffusion_functions : dict
            Diffusion function for each variable
        jump_intensities : dict, optional
            Jump intensities for discrete shocks
        """
        self.variable_names = variable_names
        self.drift_functions = drift_functions
        self.diffusion_functions = diffusion_functions
        self.jump_intensities = jump_intensities or {}
        self.dim = len(variable_names)

    def simulate(
        self,
        x0: Dict[str, float],
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Simulate geopolitical dynamics.

        Parameters
        ----------
        x0 : dict
            Initial conditions {variable: value}
        T : float
            Final time
        dt : float
            Time step
        n_paths : int
            Number of paths

        Returns
        -------
        dict
            Simulated trajectories {variable: array}
        """
        # Convert to array
        x0_array = np.array([x0[var] for var in self.variable_names])

        # Time grid
        n_steps = int(T / dt)
        t = np.linspace(0, T, n_steps + 1)

        # Storage
        trajectories = {var: np.zeros((n_paths, n_steps + 1)) for var in self.variable_names}

        # Initialize
        for i, var in enumerate(self.variable_names):
            trajectories[var][:, 0] = x0_array[i]

        sqrt_dt = np.sqrt(dt)

        # Simulate
        for step in range(n_steps):
            t_current = t[step]

            for path in range(n_paths):
                # Current state
                x_current = {
                    var: trajectories[var][path, step]
                    for var in self.variable_names
                }

                # Update each variable
                for i, var in enumerate(self.variable_names):
                    # Drift
                    drift = self.drift_functions[var](x_current, t_current) * dt

                    # Diffusion
                    dW = np.random.randn() * sqrt_dt
                    diffusion = self.diffusion_functions[var](x_current, t_current) * dW

                    # Jumps
                    jump = 0.0
                    if var in self.jump_intensities:
                        n_jumps = poisson.rvs(self.jump_intensities[var] * dt)
                        if n_jumps > 0:
                            # Simple additive jump
                            jump = np.random.normal(0, 0.1) * n_jumps

                    # Update
                    new_value = x_current[var] + drift + diffusion + jump

                    # Constraints (e.g., probabilities in [0,1])
                    new_value = np.clip(new_value, 0, 1)

                    trajectories[var][path, step + 1] = new_value

        return trajectories


def ornstein_uhlenbeck_process(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    dt: float,
    n_paths: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Ornstein-Uhlenbeck process (mean-reverting).

    dx_t = θ(μ - x_t)dt + σ dW_t

    Parameters
    ----------
    theta : float
        Mean reversion speed
    mu : float
        Long-term mean
    sigma : float
        Volatility
    x0 : float
        Initial value
    T : float
        Final time
    dt : float
        Time step
    n_paths : int
        Number of paths

    Returns
    -------
    tuple
        (time_grid, paths)
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros((n_paths, n_steps + 1))
    x[:, 0] = x0

    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        dW = np.random.randn(n_paths) * sqrt_dt
        x[:, i + 1] = x[:, i] + theta * (mu - x[:, i]) * dt + sigma * dW

    return t, x
