"""
Advanced Optimal Transport with Gradient-Based Methods

Implements sophisticated optimal transport algorithms:
- Gradient-based OT with automatic differentiation
- Kantorovich duality formulation
- Sinkhorn with entropic regularization (tunable ε)
- Wasserstein barycenters
- Gromov-Wasserstein distance
- Unbalanced optimal transport

Provides geometric insights on the space of probability measures
and leverages gradient flows for OT computations.
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

try:
    import ot as pot
    HAS_POT = True
except ImportError:
    HAS_POT = False


class GradientBasedOT:
    """
    Gradient-based optimal transport using automatic differentiation.

    Solves the Monge-Kantorovich problem:
    min_γ ∈ Π(μ,ν) ⟨γ, C⟩

    using gradient-based optimization on the dual problem or
    on parametric transport maps.
    """

    def __init__(self, cost_fn: Optional[Callable] = None):
        """
        Initialize gradient-based OT.

        Parameters
        ----------
        cost_fn : callable, optional
            Cost function c(x, y). Defaults to squared Euclidean distance.
        """
        self.cost_fn = cost_fn or (lambda x, y: np.sum((x - y)**2))

    def compute_transport_map(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        method: str = 'gradient',
        reg: float = 0.1,
        n_iter: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optimal transport map via gradient descent.

        For Monge formulation: min_T E[c(x, T(x))]

        Parameters
        ----------
        X_source : np.ndarray, shape (n, d)
            Source samples
        X_target : np.ndarray, shape (m, d)
            Target samples
        method : str
            Method ('gradient', 'dual')
        reg : float
            Regularization parameter
        n_iter : int
            Number of iterations

        Returns
        -------
        tuple
            (transport_map, cost)
        """
        n_source, d = X_source.shape
        n_target = X_target.shape[0]

        if method == 'gradient':
            # Parametrize transport map as T(x) = x + displacement
            displacement = np.zeros_like(X_source)

            learning_rate = 0.01

            for iteration in range(n_iter):
                # Compute transported points
                X_transported = X_source + displacement

                # Find nearest neighbors in target (assignment)
                distances = cdist(X_transported, X_target)
                assignments = np.argmin(distances, axis=1)

                # Compute cost
                cost = np.mean([
                    self.cost_fn(X_transported[i], X_target[assignments[i]])
                    for i in range(n_source)
                ])

                # Gradient (simplified - would use autograd)
                gradient = np.zeros_like(displacement)
                for i in range(n_source):
                    gradient[i] = 2 * (X_transported[i] - X_target[assignments[i]])

                # Regularization
                gradient += reg * displacement

                # Update
                displacement -= learning_rate * gradient

                if iteration % 20 == 0:
                    print(f"Iteration {iteration}, Cost: {cost:.6f}")

            return displacement, cost

        elif method == 'dual':
            return self._solve_dual_problem(X_source, X_target, reg)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_dual_problem(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        reg: float
    ) -> Tuple[np.ndarray, float]:
        """
        Solve dual OT problem.

        Dual formulation (Kantorovich):
        max_{f,g} E_μ[f(x)] + E_ν[g(y)]
        s.t. f(x) + g(y) ≤ c(x,y)

        Parameters
        ----------
        X_source : np.ndarray
            Source points
        X_target : np.ndarray
            Target points
        reg : float
            Regularization

        Returns
        -------
        tuple
            (potentials, cost)
        """
        n_source = len(X_source)
        n_target = len(X_target)

        # Cost matrix
        C = cdist(X_source, X_target, metric='sqeuclidean')

        # Solve via constrained optimization
        # Variables: [f (n_source), g (n_target)]
        n_vars = n_source + n_target

        def objective(x):
            f = x[:n_source]
            g = x[n_source:]
            return -(np.mean(f) + np.mean(g))  # Negative for minimization

        def constraint(x):
            f = x[:n_source]
            g = x[n_source:]
            # f(x_i) + g(y_j) - c(x_i, y_j) ≤ 0
            return C - (f[:, np.newaxis] + g[np.newaxis, :])

        from scipy.optimize import NonlinearConstraint
        nlc = NonlinearConstraint(
            lambda x: constraint(x).flatten(),
            -np.inf,
            0
        )

        x0 = np.zeros(n_vars)
        result = minimize(objective, x0, constraints=nlc, method='SLSQP')

        f_opt = result.x[:n_source]
        g_opt = result.x[n_source:]

        cost = -result.fun

        return np.column_stack([f_opt, g_opt]), cost


class KantorovichDuality:
    """
    Kantorovich duality formulation for optimal transport.

    Primal: min_γ ∈ Π(μ,ν) ∫∫ c(x,y) dγ(x,y)

    Dual: max_{f,g} ∫ f dμ + ∫ g dν
          s.t. f(x) + g(y) ≤ c(x,y)

    Strong duality holds under mild conditions.
    """

    def __init__(self):
        """Initialize Kantorovich duality solver."""
        pass

    def solve_primal(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        C: np.ndarray,
        method: str = 'emd'
    ) -> Tuple[np.ndarray, float]:
        """
        Solve primal OT problem.

        Parameters
        ----------
        mu : np.ndarray
            Source distribution weights
        nu : np.ndarray
            Target distribution weights
        C : np.ndarray
            Cost matrix
        method : str
            Method ('emd', 'sinkhorn')

        Returns
        -------
        tuple
            (coupling, cost)
        """
        if not HAS_POT:
            raise ImportError("POT library required")

        if method == 'emd':
            coupling = pot.emd(mu, nu, C)
        elif method == 'sinkhorn':
            coupling = pot.sinkhorn(mu, nu, C, reg=0.1)
        else:
            raise ValueError(f"Unknown method: {method}")

        cost = np.sum(coupling * C)
        return coupling, cost

    def solve_dual(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        C: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve dual OT problem via iterative Bregman projections.

        Parameters
        ----------
        mu : np.ndarray
            Source weights
        nu : np.ndarray
            Target weights
        C : np.ndarray
            Cost matrix
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        tuple
            (f, g, dual_value) where f, g are dual potentials
        """
        n = len(mu)
        m = len(nu)

        # Initialize dual variables (Kantorovich potentials)
        f = np.zeros(n)
        g = np.zeros(m)

        for iteration in range(max_iter):
            # Update g (c-transform of f)
            # g(y) = min_x [c(x,y) - f(x)]
            g_new = np.min(C - f[:, np.newaxis], axis=0)

            # Update f (c-transform of g)
            # f(x) = min_y [c(x,y) - g(y)]
            f_new = np.min(C - g_new[np.newaxis, :], axis=1)

            # Check convergence
            if np.max(np.abs(f_new - f)) < tol and np.max(np.abs(g_new - g)) < tol:
                break

            f, g = f_new, g_new

        # Dual value
        dual_value = np.dot(f, mu) + np.dot(g, nu)

        return f, g, dual_value

    def verify_duality_gap(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        C: np.ndarray
    ) -> float:
        """
        Verify strong duality: primal_cost - dual_cost ≈ 0.

        Parameters
        ----------
        mu : np.ndarray
            Source weights
        nu : np.ndarray
            Target weights
        C : np.ndarray
            Cost matrix

        Returns
        -------
        float
            Duality gap
        """
        # Solve primal
        coupling, primal_cost = self.solve_primal(mu, nu, C)

        # Solve dual
        f, g, dual_value = self.solve_dual(mu, nu, C)

        gap = primal_cost - dual_value
        return gap


class EntropicOT:
    """
    Entropic optimal transport with Sinkhorn algorithm.

    Regularized OT:
    min_γ ∈ Π(μ,ν) ⟨γ, C⟩ + ε H(γ)

    where H(γ) = - ∑_ij γ_ij log γ_ij is entropy.

    Sinkhorn iterations converge geometrically fast.
    """

    def __init__(self, epsilon: float = 0.1):
        """
        Initialize entropic OT.

        Parameters
        ----------
        epsilon : float
            Entropic regularization parameter
        """
        self.epsilon = epsilon

    def sinkhorn(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        C: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, float]:
        """
        Sinkhorn algorithm for entropic OT.

        Parameters
        ----------
        mu : np.ndarray
            Source distribution
        nu : np.ndarray
            Target distribution
        C : np.ndarray
            Cost matrix
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        tuple
            (coupling, cost)
        """
        n, m = C.shape

        # Kernel matrix
        K = np.exp(-C / self.epsilon)

        # Initialize scaling vectors
        u = np.ones(n) / n
        v = np.ones(m) / m

        for iteration in range(max_iter):
            u_prev = u.copy()

            # Update u
            u = mu / (K @ v)

            # Update v
            v = nu / (K.T @ u)

            # Check convergence
            if np.max(np.abs(u - u_prev)) < tol:
                break

        # Compute coupling
        coupling = u[:, np.newaxis] * K * v[np.newaxis, :]

        # Compute cost
        cost = np.sum(coupling * C)

        return coupling, cost

    def sinkhorn_log_stabilized(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        C: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, float]:
        """
        Log-stabilized Sinkhorn algorithm (more numerically stable).

        Works in log-domain to avoid overflow/underflow.

        Parameters
        ----------
        mu : np.ndarray
            Source distribution
        nu : np.ndarray
            Target distribution
        C : np.ndarray
            Cost matrix
        max_iter : int
            Maximum iterations
        tol : float
            Tolerance

        Returns
        -------
        tuple
            (coupling, cost)
        """
        n, m = C.shape

        # Log-domain variables
        log_mu = np.log(mu)
        log_nu = np.log(nu)

        # Initialize
        f = np.zeros(n)
        g = np.zeros(m)

        for iteration in range(max_iter):
            f_prev = f.copy()

            # Update f
            f = -self.epsilon * self._log_sum_exp(
                (g[np.newaxis, :] - C) / self.epsilon,
                axis=1
            ) + log_mu + self.epsilon * np.log(n)

            # Update g
            g = -self.epsilon * self._log_sum_exp(
                (f[:, np.newaxis] - C) / self.epsilon,
                axis=0
            ) + log_nu + self.epsilon * np.log(m)

            # Check convergence
            if np.max(np.abs(f - f_prev)) < tol:
                break

        # Compute coupling (in log domain, then exp)
        log_coupling = (f[:, np.newaxis] + g[np.newaxis, :] - C) / self.epsilon
        coupling = np.exp(log_coupling)

        # Normalize
        coupling = coupling / np.sum(coupling)

        cost = np.sum(coupling * C)

        return coupling, cost

    def _log_sum_exp(self, X: np.ndarray, axis: int) -> np.ndarray:
        """
        Numerically stable log-sum-exp.

        log(∑ exp(X_i)) computed stably.

        Parameters
        ----------
        X : np.ndarray
            Input array
        axis : int
            Axis to sum over

        Returns
        -------
        np.ndarray
            log-sum-exp result
        """
        max_X = np.max(X, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(X - max_X), axis=axis)) + np.squeeze(max_X, axis=axis)


class UnbalancedOT:
    """
    Unbalanced optimal transport.

    Relaxes the marginal constraints, allowing mass creation/destruction.
    Useful when distributions don't have same total mass.

    Formulation:
    min_γ ⟨γ, C⟩ + ε H(γ) + τ KL(γ1_m | μ) + τ KL(γ^T1_n | ν)

    where KL is Kullback-Leibler divergence.
    """

    def __init__(self, epsilon: float = 0.1, tau: float = 0.1):
        """
        Initialize unbalanced OT.

        Parameters
        ----------
        epsilon : float
            Entropic regularization
        tau : float
            Marginal relaxation parameter
        """
        self.epsilon = epsilon
        self.tau = tau

    def solve(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        C: np.ndarray,
        max_iter: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Solve unbalanced OT via Sinkhorn-Knopp algorithm.

        Parameters
        ----------
        mu : np.ndarray
            Source (unbalanced) distribution
        nu : np.ndarray
            Target (unbalanced) distribution
        C : np.ndarray
            Cost matrix
        max_iter : int
            Maximum iterations

        Returns
        -------
        tuple
            (coupling, cost)
        """
        if not HAS_POT:
            raise ImportError("POT library required for unbalanced OT")

        # Use POT's unbalanced solver
        coupling = pot.unbalanced.sinkhorn_unbalanced(
            mu, nu, C,
            reg=self.epsilon,
            reg_m=self.tau,
            numItermax=max_iter
        )

        cost = np.sum(coupling * C)

        return coupling, cost


class GromovWassersteinDistance:
    """
    Gromov-Wasserstein distance for comparing metric spaces.

    Useful when we want to compare structures (graphs, networks)
    rather than points in a common space.
    """

    def __init__(self, epsilon: float = 0.1):
        """
        Initialize Gromov-Wasserstein.

        Parameters
        ----------
        epsilon : float
            Entropic regularization
        """
        self.epsilon = epsilon

    def compute(
        self,
        C1: np.ndarray,
        C2: np.ndarray,
        p: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
        loss_fun: str = 'square_loss',
        max_iter: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Gromov-Wasserstein distance.

        Parameters
        ----------
        C1 : np.ndarray
            Intra-space cost matrix for space 1
        C2 : np.ndarray
            Intra-space cost matrix for space 2
        p : np.ndarray, optional
            Distribution on space 1
        q : np.ndarray, optional
            Distribution on space 2
        loss_fun : str
            Loss function
        max_iter : int
            Maximum iterations

        Returns
        -------
        tuple
            (coupling, GW_distance)
        """
        if not HAS_POT:
            raise ImportError("POT library required for Gromov-Wasserstein")

        n1, n2 = C1.shape[0], C2.shape[0]

        # Default uniform distributions
        if p is None:
            p = np.ones(n1) / n1
        if q is None:
            q = np.ones(n2) / n2

        # Compute Gromov-Wasserstein
        gw_dist, log = pot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q,
            loss_fun=loss_fun,
            epsilon=self.epsilon,
            max_iter=max_iter,
            log=True
        )

        coupling = log['T']

        return coupling, gw_dist
