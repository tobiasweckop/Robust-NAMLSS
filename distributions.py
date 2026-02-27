import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Gamma as torch_gamma
from torch.distributions import Normal as torch_normal


class Distribution:
    registry = {}

    def __init_subclass__(cls, **kwargs):
        
        # This currently does nothing
        super().__init_subclass__(**kwargs)

        # Registers the subclass in the registry
        Distribution.registry[cls.__name__.lower()] = cls



class Normal(Distribution):

    @classmethod
    def transform(cls, parameters):
        mu = parameters[:, 0]
        sigma = parameters[:, 1]
        return mu, sigma

    @classmethod
    def nll_loss(cls, parameters, y, c = None):

        mu = parameters[:, 0]
        sigma = parameters[:, 1]

        normal_dist = torch_normal(mu, sigma)
        log_likelihood = normal_dist.log_prob(y)

        if c is not None:
            log_likelihood = torch.log((1 + torch.exp(log_likelihood + c)) / (1 + torch.exp(c)))

        nll = -log_likelihood.mean()
        return nll

    @classmethod
    def get_param_count(cls):
        return 2



class Gamma(Distribution):

    @classmethod
    def get_param_count(cls):
        return 2

    @classmethod
    def transform(cls, stacked_tensor):
        alpha = stacked_tensor[:, :, 0]
        theta = stacked_tensor[:, :, 1]

        alpha = F.softplus(alpha)
        theta = F.softplus(theta)

        transformed_tensor = torch.stack([alpha, theta], dim=2)

        return transformed_tensor

    @classmethod
    def nll_loss(cls, parameter_tensor, y, c = None):

        alpha = parameter_tensor[:, 0]
        theta = parameter_tensor[:, 1]

        gamma_dist = torch_gamma(alpha, 1/theta)                  # Note: Pytorch Gamma uses shape and rate (1/scale)
        log_likelihood = gamma_dist.log_prob(y)

        if c is not None:
            log_likelihood = torch.log((1 + torch.exp(log_likelihood + c)) / (1 + torch.exp(c)))

        nll = -log_likelihood.mean()
        return nll

    @classmethod
    def cdf(cls, parameter_tensor, y):

        alpha = parameter_tensor[:, 0]
        theta = parameter_tensor[:, 1]

        dist = torch_gamma(alpha, 1/theta)
        y_cdf = dist.cdf(y).squeeze()

        return y_cdf



class BCCG(Distribution):

    @classmethod
    def get_param_count(cls):
        return 3
    
    @classmethod
    def transform(cls, stacked_tensor):
        mu = stacked_tensor[:, :, 0]
        sigma = stacked_tensor[:, :, 1]
        nu = stacked_tensor[:, :, 2]

        mu = F.softplus(mu)
        sigma = F.softplus(sigma)
        nu = nu

        transformed_tensor = torch.stack([mu, sigma, nu], dim=2)

        return transformed_tensor

    @classmethod
    def nll_loss(cls, parameter_tensor, y, c = None):
        
        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]
        nu = parameter_tensor[:, 2]
        
        normal = torch_normal(0, 1)
        z = torch.where(nu == 0, 1/sigma * torch.log(y/mu), 1/(sigma * nu) * ((y/mu)**nu - 1))

        log_likelihood = (nu - 1) * torch.log(y) - 1/2 * z**2 - nu * torch.log(mu) - torch.log(sigma) - 1/2 * torch.log(torch.tensor(2*torch.pi)) - torch.log(normal.cdf(1/(sigma * torch.abs(nu))))

        if c is not None:
            log_likelihood = torch.log((1 + torch.exp(log_likelihood + c)) / (1 + torch.exp(c)))
        
        nll = -log_likelihood.mean()
        return nll

    @classmethod
    def cdf(cls, parameter_tensor, y):

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]
        nu = parameter_tensor[:, 2]

        normal = torch_normal(0, 1)
        z = torch.where(nu == 0, 1/sigma * torch.log(y/mu), 1/(sigma * nu) * ((y/mu)**nu - 1))

        y_cdf = normal.cdf(z)

        return y_cdf

    @classmethod
    def icdf(cls, parameter_tensor, p):

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]
        nu = parameter_tensor[:, 2]

        normal = torch_normal(0, 1)
        z = normal.icdf(p)

        y_icdf = torch.where(nu == 0, mu * torch.exp(sigma * z), mu * (1 + sigma * nu * z)**(1/nu))

        return y_icdf