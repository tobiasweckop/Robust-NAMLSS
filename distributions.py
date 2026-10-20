import math
import torch
import numpy as np

from scipy import stats
from scipy.integrate import quad

import torch.nn.functional as F
from torch.distributions import Gamma as torch_gamma
from torch.distributions import Normal as torch_normal


class Distribution:
    registry = {}

    def __init_subclass__(cls, **kwargs):
        
        # Only here because of conventions
        super().__init_subclass__(**kwargs)

        # Registers the subclass in the registry
        Distribution.registry[cls.__name__.lower()] = cls


    @classmethod
    def _verify_input_shapes(cls, parameter_tensor, vector = None):

        if parameter_tensor.ndim != 2:
            raise ValueError(
                f"Expected parameter_tensor to have shape (N,{cls.get_param_count()}). Got {tuple(parameter_tensor.shape)} instead."
            )

        if parameter_tensor.shape[1] != cls.get_param_count():
            raise ValueError(
                f"Expected {cls.get_param_count()} parameters. Got {parameter_tensor.shape[1]} instead."
            )

        if not isinstance(parameter_tensor, torch.Tensor):
            raise TypeError("parameter_tensor must be a torch.Tensor.")

        if vector is not None:
            if vector.ndim != 1:
                raise ValueError(
                    f"Expected vector to have shape (N,), got {tuple(vector.shape)}."
                )

            if parameter_tensor.shape[0] != vector.shape[0]:
                raise ValueError("Batch sizes do not match.")
                



class Normal(Distribution):

    @classmethod
    def get_param_count(cls):
        return 2


    @classmethod
    def transform(cls, stacked_parameter_tensor):

        mu = stacked_parameter_tensor[:, :, 0]
        sigma = F.softplus(stacked_parameter_tensor[:, :, 1])

        transformed_tensor = torch.stack([mu, sigma], dim=2)

        return transformed_tensor


    @classmethod
    def pdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        y_pdf = 1/(math.sqrt(2 * torch.pi) * sigma) * torch.exp(-(y - mu)**2 / (2 * sigma**2))

        return y_pdf


    @classmethod
    def log_likelihood(cls, parameter_tensor, y): # can be more numerically stable than pdf

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]
        
        return (-0.5 * torch.log(2 * torch.pi) - torch.log(sigma) - 0.5 * ((y - mu) / sigma)**2)


    @classmethod
    def nll_loss(cls, parameter_tensor, y, c = None, observation_weights = None):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        normal_dist = torch_normal(mu, sigma)
        log_likelihood = normal_dist.log_prob(y)

        # Calculate robust log-likelihood if c is given
        if c is not None:
            log_likelihood = torch.log((1 + torch.exp(log_likelihood + c)) / (1 + torch.exp(c)))

        # Apply observation weights, if provided
        if observation_weights is not None:
            if not isinstance(observation_weights, torch.Tensor):
                observation_weights = torch.tensor(observation_weights)

            if observation_weights.ndim != 1:
                raise ValueError(f"Expected observation_weights to have shape (N,), got {tuple(observation_weights.shape)}.")
            if parameter_tensor.shape[0] != observation_weights.shape[0]:
                raise ValueError("Batch sizes do not match.")

            log_likelihood = log_likelihood * observation_weights

        nll = -log_likelihood.mean()
        return nll


    @classmethod
    def cdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        normal_dist = torch_normal(mu, sigma)
        y_cdf = normal_dist.cdf(y)

        return y_cdf


    @classmethod
    def icdf(cls, parameter_tensor, p):

        cls._verify_input_shapes(parameter_tensor, p)

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        normal_dist = torch_normal(mu, sigma)
        y_icdf = normal_dist.icdf(p)

        return y_icdf

    
    @classmethod
    def sample(cls, parameter_tensor):

        cls._verify_input_shapes(parameter_tensor)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        normal_dist = torch_normal(mu, sigma)
        samples = normal_dist.sample()

        return samples



class LogNormal(Distribution):

    standard_normal = torch_normal(0, 1)


    @classmethod
    def get_param_count(cls):
        return 2


    @classmethod
    def transform(cls, stacked_parameter_tensor):

        mu = stacked_parameter_tensor[:, :, 0]
        sigma = stacked_parameter_tensor[:, :, 1]

        mu = mu
        sigma = F.softplus(sigma)

        transformed_tensor = torch.stack([mu, sigma], dim=2)

        return transformed_tensor

    
    @classmethod
    def pdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        y_pdf = 1/(torch.sqrt(2 * torch.pi * sigma**2)) * 1/y * torch.exp(-(torch.log(y) - mu)**2 / (2 * sigma**2))

        return y_pdf


    @classmethod
    def nll_loss(cls, parameter_tensor, y, c = None, observation_weights = None):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        log_likelihood = (
            - torch.log(y)
            - torch.log(sigma)
            - 0.5 * torch.log(torch.tensor(2 * torch.pi))
            - (torch.log(y) - mu)**2 / (2 * sigma**2)
        )

        # Calculate robust log-likelihood if c is given
        if c is not None:
            log_likelihood = torch.log((1 + torch.exp(log_likelihood + c)) / (1 + torch.exp(c)))

        # Apply observation weights, if provided
        if observation_weights is not None:
            if not isinstance(observation_weights, torch.Tensor):
                observation_weights = torch.tensor(observation_weights)

            if observation_weights.ndim != 1:
                raise ValueError(f"Expected observation_weights to have shape (N,), got {tuple(observation_weights.shape)}.")
            if parameter_tensor.shape[0] != observation_weights.shape[0]:
                raise ValueError("Batch sizes do not match.")

            log_likelihood = log_likelihood * observation_weights

        nll = -log_likelihood.mean()
        return nll

    
    @classmethod
    def cdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        y_cdf = cls.standard_normal.cdf((torch.log(y) - mu) / sigma)

        return y_cdf

    
    @classmethod
    def icdf(cls, parameter_tensor, p):

        cls._verify_input_shapes(parameter_tensor, p)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        z_p = cls.standard_normal.icdf(p)
        y_icdf = torch.exp(mu + sigma * z_p)

        return y_icdf

    
    @classmethod
    def sample(cls, parameter_tensor):

        cls._verify_input_shapes(parameter_tensor)

        sample_size = parameter_tensor.shape[0]
        probabilities = torch.rand(sample_size)

        return cls.icdf(parameter_tensor, probabilities)


class Gamma(Distribution):

    @classmethod
    def get_param_count(cls):
        return 2


    @classmethod
    def transform(cls, stacked_parameter_tensor):

        alpha = stacked_parameter_tensor[:, :, 0]
        theta = stacked_parameter_tensor[:, :, 1]

        alpha = F.softplus(alpha)
        theta = F.softplus(theta)

        transformed_tensor = torch.stack([alpha, theta], dim=2)

        return transformed_tensor


    @classmethod
    def pdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)

        alpha = parameter_tensor[:, 0]
        theta = parameter_tensor[:, 1]

        gamma_dist = torch_gamma(alpha, 1/theta) # Note: Pytorch Gamma uses shape and rate (1/theta)

        y_pdf = gamma_dist.log_prob(y).exp()

        return y_pdf


    @classmethod
    def nll_loss(cls, parameter_tensor, y, c = None, observation_weights = None):

        cls._verify_input_shapes(parameter_tensor, y)

        alpha = parameter_tensor[:, 0]
        theta = parameter_tensor[:, 1]

        gamma_dist = torch_gamma(alpha, 1/theta) # Note: Pytorch Gamma uses shape and rate (1/theta)
        log_likelihood = gamma_dist.log_prob(y)

        # Calculate robust log-likelihood if c is given
        if c is not None:
            log_likelihood = torch.log((1 + torch.exp(log_likelihood + c)) / (1 + torch.exp(c)))

        # Apply observation weights, if provided
        if observation_weights is not None:
            if not isinstance(observation_weights, torch.Tensor):
                observation_weights = torch.tensor(observation_weights)

            if observation_weights.ndim != 1:
                raise ValueError(f"Expected observation_weights to have shape (N,), got {tuple(observation_weights.shape)}.")
            if parameter_tensor.shape[0] != observation_weights.shape[0]:
                raise ValueError("Batch sizes do not match.")

            log_likelihood = log_likelihood * observation_weights

        nll = -log_likelihood.mean()
        return nll


    @classmethod
    def cdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)

        alpha = parameter_tensor[:, 0]
        theta = parameter_tensor[:, 1]

        dist = torch_gamma(alpha, 1/theta)
        y_cdf = dist.cdf(y)

        return y_cdf

    
    @classmethod
    def icdf(cls, parameter_tensor, p):

        cls._verify_input_shapes(parameter_tensor, p)

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p)

        alpha = parameter_tensor[:, 0]
        theta = parameter_tensor[:, 1]

        y_icdf = stats.gamma.ppf(p, a = alpha, scale = theta)
        y_icdf = torch.tensor(y_icdf)

        return y_icdf


    @classmethod
    def sample(cls, parameter_tensor):

        cls._verify_input_shapes(parameter_tensor)

        sample_size = parameter_tensor.shape[0]
        probabilities = torch.rand(sample_size)

        return cls.icdf(parameter_tensor, probabilities)



class BCCG(Distribution):
    
    standard_normal = torch_normal(0, 1)

    @classmethod
    def get_param_count(cls):
        return 3

    
    @classmethod
    def transform(cls, stacked_parameter_tensor):

        mu = stacked_parameter_tensor[:, :, 0]
        sigma = stacked_parameter_tensor[:, :, 1]
        nu = stacked_parameter_tensor[:, :, 2]

        mu = F.softplus(mu)
        sigma = F.softplus(sigma)
        nu = nu

        transformed_tensor = torch.stack([mu, sigma, nu], dim=2)

        return transformed_tensor


    @classmethod
    def pdf(cls, parameter_tensor, y):

        cls._verify_input_shapes(parameter_tensor, y)


    @classmethod
    def nll_loss(cls, parameter_tensor, y, c = None, observation_weights = None):

        cls._verify_input_shapes(parameter_tensor, y)

        mu = parameter_tensor[:,0]
        sigma = parameter_tensor[:,1]
        nu = parameter_tensor[:,2]

        z = torch.empty_like(y)
        z[nu != 0] = ((((y[nu != 0] / mu[nu != 0])**nu[nu != 0])-1) / (nu[nu != 0] * sigma[nu != 0]))
        z[nu == 0] = torch.log(y[nu == 0]/mu[nu == 0])/sigma[nu == 0]

        log_phi = torch.zeros_like(nu)
        log_phi[nu != 0] = torch.special.log_ndtr(1/(sigma[nu != 0] * torch.abs(nu[nu != 0])))

        log_likelihood = nu * torch.log(y/mu) - torch.log(sigma) - (z*z)/2 -torch.log(y) -(math.log(2*math.pi))/2 - log_phi

        if c is not None:
            log_likelihood = torch.nn.functional.softplus(log_likelihood + c) - torch.nn.functional.softplus(c)

        if observation_weights is not None:
            log_likelihood = log_likelihood * observation_weights

        nll = - log_likelihood.mean()

        return nll


    @classmethod
    def cdf(cls, parameter_tensor, q):

        #TODO: handle q == 0 correctly. I really dont want to implement this right now, but it should be done at some point.

        cls._verify_input_shapes(parameter_tensor)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]
        nu = parameter_tensor[:, 2]

        q = torch.as_tensor(q)

        # Allow a single q or one q per parameter row
        if q.ndim == 0:
            q = q.expand_as(mu)
        elif q.ndim != 1 or q.shape != mu.shape:
            raise ValueError("q must be a scalar or a 1D tensor matching parameter_tensor")
        if torch.any(mu <= 0):
            raise ValueError("mu must be positive")
        if torch.any(sigma <= 0):
            raise ValueError("sigma must be positive")
        if torch.any(q < 0):
            raise ValueError("q must be non-negative")

        # Handle q == 0 by slightly elevating it, then manually assigning it zero at the end of calculations
        positive_q = q > 0
        q = q.clamp_min(torch.finfo(q.dtype).tiny)

        z = torch.empty_like(mu)
        z[nu != 0] = (((q[nu != 0] / mu[nu != 0])**nu[nu != 0] - 1) / (nu[nu != 0] * sigma[nu != 0]))
        z[nu == 0] = torch.log(q[nu == 0] / mu[nu == 0]) / sigma[nu == 0]

        phi = torch.special.ndtr(z)

        a = torch.zeros_like(nu)
        a[nu > 0] = torch.special.ndtr(-1/(sigma[nu > 0] * torch.abs(nu[nu > 0])))

        b = torch.ones_like(nu)
        b[nu != 0] = torch.special.ndtr(1/(sigma[nu != 0] * torch.abs(nu[nu != 0])))

        fY = (phi - a)/b
        fY = torch.where(positive_q, fY, 0.0)

        return fY


    @classmethod
    def icdf(cls, parameter_tensor, p):

        '''Takes a cumulative probability and returns the associated y-value.'''

        cls._verify_input_shapes(parameter_tensor, p)

        mu = parameter_tensor[:,0]
        sigma = parameter_tensor[:,1]
        nu = parameter_tensor[:,2]

        p = torch.as_tensor(p)

        if p.ndim == 0:
            p = p.expand_as(mu)
        elif p.shape != mu.shape:
            raise ValueError("p must be a scalar or have shape (N,).")
        if torch.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be in [0, 1].")

        z = torch.empty_like(nu)
        z[nu <= 0] = cls.standard_normal.icdf(p[nu <= 0] * cls.standard_normal.cdf(1/(sigma[nu <= 0] * torch.abs(nu[nu <= 0]))))
        z[nu > 0] = cls.standard_normal.icdf(1 - (1 - p[nu > 0]) * cls.standard_normal.cdf(1/(sigma[nu > 0] * torch.abs(nu[nu > 0]))))

        y = torch.empty_like(nu)
        y[nu != 0] = mu[nu != 0] * ((nu[nu != 0] * sigma[nu != 0] * z[nu != 0] + 1)**(1/nu[nu != 0]))
        y[nu == 0] = mu[nu == 0] * torch.exp(sigma[nu == 0] * z[nu == 0])

        y[p == 0] = 0.0
        y[p == 1] = float("inf")

        return y



    @classmethod
    def sample(cls, parameter_tensor):

        cls._verify_input_shapes(parameter_tensor)
