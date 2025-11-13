"""
Variational Inference (VI) Engine

Implements scalable approximate Bayesian inference via optimization:
- Mean-field variational inference
- Automatic Differentiation Variational Inference (ADVI)
- Evidence Lower Bound (ELBO) optimization
- Coordinate ascent variational inference (CAVI)

Provides high-dimensional posterior approximation when MCMC is intractable.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize


@dataclass
class VariationalDistribution:
    """
    Parametric variational distribution q(z | λ).

    Attributes
    ----------
    family : str
        Distribution family ('normal', 'multivariate_normal')
    parameters : dict
        Distribution parameters
    """
    family: str
    parameters: Dict[str, np.ndarray]

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from variational distribution."""
        if self.family == 'normal':
            mu = self.parameters['mu']
            sigma = self.parameters['sigma']
            return np.random.normal(mu, sigma, size=(n_samples, len(mu)))
        elif self.family == 'multivariate_normal':
            mu = self.parameters['mu']
            cov = self.parameters['cov']
            return np.random.multivariate_normal(mu, cov, size=n_samples)
        else:
            raise ValueError(f"Unknown family: {self.family}")

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        if self.family == 'normal':
            mu = self.parameters['mu']
            sigma = self.parameters['sigma']
            return np.sum(norm.logpdf(z, loc=mu, scale=sigma), axis=-1)
        elif self.family == 'multivariate_normal':
            mu = self.parameters['mu']
            cov = self.parameters['cov']
            return multivariate_normal.logpdf(z, mean=mu, cov=cov)
        else:
            raise ValueError(f"Unknown family: {self.family}")

    def entropy(self) -> float:
        """Compute entropy H[q]."""
        if self.family == 'normal':
            sigma = self.parameters['sigma']
            # H = 0.5 * log(2πeσ²)
            return 0.5 * np.sum(np.log(2 * np.pi * np.e * sigma**2))
        elif self.family == 'multivariate_normal':
            cov = self.parameters['cov']
            d = len(cov)
            # H = 0.5 * log((2πe)^d |Σ|)
            sign, logdet = np.linalg.slogdet(cov)
            return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
        else:
            raise ValueError(f"Unknown family: {self.family}")


class VariationalInference:
    """
    Variational Inference engine.

    Approximates posterior p(z|x) with variational distribution q(z|λ)
    by maximizing Evidence Lower Bound (ELBO):

    ELBO(λ) = E_q[log p(x,z)] - E_q[log q(z|λ)]
            = E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z|λ)]

    Equivalently: minimize KL(q(z|λ) || p(z|x))
    """

    def __init__(
        self,
        log_joint: Callable,
        variational_family: str = 'normal',
        n_samples: int = 100
    ):
        """
        Initialize variational inference.

        Parameters
        ----------
        log_joint : callable
            Log joint probability: log p(x, z)
        variational_family : str
            Variational family ('normal', 'multivariate_normal')
        n_samples : int
            Number of Monte Carlo samples for ELBO estimation
        """
        self.log_joint = log_joint
        self.variational_family = variational_family
        self.n_samples = n_samples
        self.q = None

    def elbo(
        self,
        variational_params: np.ndarray,
        param_shapes: Dict[str, Tuple],
        observed_data: Any
    ) -> float:
        """
        Compute Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(x,z)] - E_q[log q(z)]

        Parameters
        ----------
        variational_params : np.ndarray
            Flattened variational parameters
        param_shapes : dict
            Shapes of each parameter
        observed_data : any
            Observed data x

        Returns
        -------
        float
            ELBO value
        """
        # Unpack parameters
        params = self._unpack_params(variational_params, param_shapes)

        # Create variational distribution
        q = VariationalDistribution(self.variational_family, params)

        # Sample from q
        z_samples = q.sample(self.n_samples)

        # Compute E_q[log p(x, z)]
        log_joint_vals = np.array([self.log_joint(z, observed_data) for z in z_samples])
        expected_log_joint = np.mean(log_joint_vals)

        # Compute E_q[log q(z)]
        log_q_vals = q.log_prob(z_samples)
        expected_log_q = np.mean(log_q_vals)

        # ELBO
        elbo_val = expected_log_joint - expected_log_q

        return elbo_val

    def neg_elbo(self, variational_params: np.ndarray, param_shapes: Dict, observed_data: Any) -> float:
        """Negative ELBO for minimization."""
        return -self.elbo(variational_params, param_shapes, observed_data)

    def fit(
        self,
        observed_data: Any,
        init_params: Dict[str, np.ndarray],
        max_iter: int = 1000,
        method: str = 'L-BFGS-B'
    ) -> VariationalDistribution:
        """
        Fit variational distribution via ELBO optimization.

        Parameters
        ----------
        observed_data : any
            Observed data
        init_params : dict
            Initial variational parameters
        max_iter : int
            Maximum optimization iterations
        method : str
            Optimization method

        Returns
        -------
        VariationalDistribution
            Optimized variational distribution
        """
        # Pack initial parameters
        flat_params, param_shapes = self._pack_params(init_params)

        # Optimize
        result = minimize(
            fun=self.neg_elbo,
            x0=flat_params,
            args=(param_shapes, observed_data),
            method=method,
            options={'maxiter': max_iter, 'disp': True}
        )

        # Unpack optimized parameters
        opt_params = self._unpack_params(result.x, param_shapes)

        # Create variational distribution
        self.q = VariationalDistribution(self.variational_family, opt_params)

        return self.q

    def _pack_params(self, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """Pack parameters into flat array."""
        flat = []
        shapes = {}
        for key, val in params.items():
            flat.append(val.flatten())
            shapes[key] = val.shape
        return np.concatenate(flat), shapes

    def _unpack_params(self, flat: np.ndarray, shapes: Dict) -> Dict[str, np.ndarray]:
        """Unpack flat array into parameters."""
        params = {}
        idx = 0
        for key, shape in shapes.items():
            size = np.prod(shape)
            params[key] = flat[idx:idx+size].reshape(shape)
            idx += size
        return params


class MeanFieldVI(VariationalInference):
    """
    Mean-Field Variational Inference.

    Assumes variational distribution factorizes:
    q(z | λ) = ∏_i q_i(z_i | λ_i)

    Uses coordinate ascent variational inference (CAVI) to optimize
    each factor in turn.
    """

    def __init__(
        self,
        log_joint: Callable,
        factor_families: List[str],
        n_samples: int = 100
    ):
        """
        Initialize mean-field VI.

        Parameters
        ----------
        log_joint : callable
            Log joint probability
        factor_families : list
            Distribution family for each factor
        n_samples : int
            Number of samples for ELBO
        """
        super().__init__(log_joint, 'mean_field', n_samples)
        self.factor_families = factor_families
        self.n_factors = len(factor_families)

    def fit_cavi(
        self,
        observed_data: Any,
        init_params: List[Dict[str, np.ndarray]],
        max_iter: int = 100,
        tol: float = 1e-4
    ) -> List[VariationalDistribution]:
        """
        Fit using Coordinate Ascent Variational Inference (CAVI).

        Parameters
        ----------
        observed_data : any
            Observed data
        init_params : list
            Initial parameters for each factor
        max_iter : int
            Maximum CAVI iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        list
            List of optimized factor distributions
        """
        # Initialize factors
        factors = [
            VariationalDistribution(family, params)
            for family, params in zip(self.factor_families, init_params)
        ]

        prev_elbo = -np.inf

        for iteration in range(max_iter):
            # Update each factor in turn
            for i in range(self.n_factors):
                # Update factor i holding others fixed
                factors[i] = self._update_factor(i, factors, observed_data)

            # Compute ELBO
            current_elbo = self._compute_mean_field_elbo(factors, observed_data)

            # Check convergence
            if abs(current_elbo - prev_elbo) < tol:
                print(f"CAVI converged at iteration {iteration}")
                break

            prev_elbo = current_elbo

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, ELBO: {current_elbo:.4f}")

        self.factors = factors
        return factors

    def _update_factor(
        self,
        factor_idx: int,
        factors: List[VariationalDistribution],
        observed_data: Any
    ) -> VariationalDistribution:
        """
        Update a single factor via optimization.

        Parameters
        ----------
        factor_idx : int
            Index of factor to update
        factors : list
            Current factor distributions
        observed_data : any
            Observed data

        Returns
        -------
        VariationalDistribution
            Updated factor
        """
        # This is a simplified version - full implementation would compute
        # conditional expectations analytically for conjugate models

        # For now, use gradient-based optimization
        current_params = factors[factor_idx].parameters

        def factor_neg_elbo(params_flat):
            # Unpack
            if self.factor_families[factor_idx] == 'normal':
                d = len(params_flat) // 2
                mu = params_flat[:d]
                log_sigma = params_flat[d:]
                sigma = np.exp(log_sigma)
                params = {'mu': mu, 'sigma': sigma}
            else:
                raise NotImplementedError

            # Create trial factor
            trial_factor = VariationalDistribution(self.factor_families[factor_idx], params)

            # Replace in factors
            trial_factors = factors.copy()
            trial_factors[factor_idx] = trial_factor

            # Compute ELBO
            elbo = self._compute_mean_field_elbo(trial_factors, observed_data)
            return -elbo

        # Pack current params
        if self.factor_families[factor_idx] == 'normal':
            params_flat = np.concatenate([
                current_params['mu'],
                np.log(current_params['sigma'])
            ])
        else:
            raise NotImplementedError

        # Optimize
        result = minimize(factor_neg_elbo, params_flat, method='L-BFGS-B')

        # Unpack
        if self.factor_families[factor_idx] == 'normal':
            d = len(result.x) // 2
            mu = result.x[:d]
            sigma = np.exp(result.x[d:])
            opt_params = {'mu': mu, 'sigma': sigma}
        else:
            raise NotImplementedError

        return VariationalDistribution(self.factor_families[factor_idx], opt_params)

    def _compute_mean_field_elbo(
        self,
        factors: List[VariationalDistribution],
        observed_data: Any
    ) -> float:
        """
        Compute ELBO for mean-field approximation.

        Parameters
        ----------
        factors : list
            Factor distributions
        observed_data : any
            Observed data

        Returns
        -------
        float
            ELBO
        """
        # Sample from each factor
        samples = []
        for factor in factors:
            samples.append(factor.sample(self.n_samples))

        # Combine samples
        z_samples = np.column_stack(samples)

        # Compute E_q[log p(x, z)]
        log_joint_vals = np.array([self.log_joint(z, observed_data) for z in z_samples])
        expected_log_joint = np.mean(log_joint_vals)

        # Compute E_q[log q(z)] = sum_i E_q[log q_i(z_i)]
        expected_log_q = 0.0
        for i, factor in enumerate(factors):
            log_q_vals = factor.log_prob(samples[i])
            expected_log_q += np.mean(log_q_vals)

        return expected_log_joint - expected_log_q


class ADVI:
    """
    Automatic Differentiation Variational Inference (ADVI).

    Transforms constrained latent variables to unconstrained space,
    then performs VI with Gaussian variational family.

    Uses reparameterization trick for low-variance gradient estimates.
    """

    def __init__(
        self,
        log_joint: Callable,
        transform_fn: Optional[Callable] = None,
        inverse_transform_fn: Optional[Callable] = None
    ):
        """
        Initialize ADVI.

        Parameters
        ----------
        log_joint : callable
            Log joint in original (possibly constrained) space
        transform_fn : callable, optional
            Transform to unconstrained space
        inverse_transform_fn : callable, optional
            Inverse transform
        """
        self.log_joint = log_joint
        self.transform_fn = transform_fn or (lambda x: x)
        self.inverse_transform_fn = inverse_transform_fn or (lambda x: x)

    def fit(
        self,
        observed_data: Any,
        latent_dim: int,
        n_samples: int = 10,
        max_iter: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit ADVI using gradient ascent on ELBO.

        Parameters
        ----------
        observed_data : any
            Observed data
        latent_dim : int
            Dimension of latent variables
        n_samples : int
            Number of samples for ELBO gradient estimation
        max_iter : int
            Maximum iterations
        learning_rate : float
            Learning rate for gradient ascent

        Returns
        -------
        tuple
            (mean, log_std) of variational distribution
        """
        # Initialize variational parameters (Gaussian in unconstrained space)
        mu = np.zeros(latent_dim)
        log_sigma = np.zeros(latent_dim)

        for iteration in range(max_iter):
            # Sample from standard normal
            epsilon = np.random.randn(n_samples, latent_dim)

            # Reparameterization: z = μ + σ * ε
            sigma = np.exp(log_sigma)
            z_unconstrained = mu + sigma * epsilon

            # Transform to constrained space
            z_constrained = np.array([self.inverse_transform_fn(z) for z in z_unconstrained])

            # Compute log joint
            log_joints = np.array([self.log_joint(z, observed_data) for z in z_constrained])

            # Compute ELBO (with entropy)
            entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * sigma**2))
            elbo = np.mean(log_joints) + entropy

            # Gradient estimates (simplified - would use autograd in practice)
            grad_mu = np.mean((log_joints[:, np.newaxis] - elbo) * (z_unconstrained - mu) / (sigma**2), axis=0)
            grad_log_sigma = np.mean(
                (log_joints[:, np.newaxis] - elbo) * ((z_unconstrained - mu)**2 / sigma**2 - 1),
                axis=0
            )

            # Update parameters
            mu = mu + learning_rate * grad_mu
            log_sigma = log_sigma + learning_rate * grad_log_sigma

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, ELBO: {elbo:.4f}")

        return mu, log_sigma
