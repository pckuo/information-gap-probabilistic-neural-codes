# for submission to ICLR 2026


# This module defines a generative model for neural population responses
# based on task priors and likelihood functions.


from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils.functions import get_Gaussian_pdf


@dataclass
class TaskPrior:
    """Represents a task prior distribution p^c(θ)"""
    name: str
    thetas: np.ndarray
    distribution: np.ndarray
    
    def __post_init__(self):
        # Normalize to ensure it's a valid probability distribution
        self.distribution = self.distribution / np.sum(self.distribution)


class LikelihoodModel:
    """Base class for likelihood models"""
    def __init__(
        self, 
        thetas: np.ndarray
    ):
        """
        Args:
            thetas: Array of possible stimulus values
        """
        self.thetas = thetas
        
    def get_likelihood_function(
        self,
        x_i
    ) -> np.ndarray:
        """Compute p(x_i|θ) for a given x_i"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_conditional_probability(
        self,
        theta_i
    ) -> np.ndarray:
        """Compute p(x|θ_i) for a given theta"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_posterior(
        self, 
        x_i, 
        prior: TaskPrior
    ) -> np.ndarray:
        """Compute p^μ(θ|x) ∝ p(x|θ) * p^μ(θ)"""
        likelihood = self.get_likelihood_function(x_i)
        posterior = likelihood * prior.distribution
        # Normalize
        posterior = posterior / np.sum(posterior)
        return posterior


class GaussianLikelihoodModel(LikelihoodModel):
    """Base class for likelihood models"""
    def __init__(
        self, 
        thetas: np.ndarray, 
        equivalent_sigma: float = 15.0,
    ):
        """
        Args:
            theta_values: Array of possible stimulus values
            likelihood_func: Function p(x|θ) that takes theta and returns likelihood
        """
        self.thetas = thetas
        self.equivalent_sigma = equivalent_sigma

    def get_likelihood_function(
        self,
        x_i
    ) -> np.ndarray:
        """Compute p(x_i|θ) for a given x_i"""
        return get_Gaussian_pdf(
            mu=x_i, 
            sigma=self.equivalent_sigma, 
            thetas=self.thetas
        )
    
    def get_conditional_probability(
        self,
        theta_i
    ) -> np.ndarray:
        """Compute p(x|θ_i) for a given theta"""
        return get_Gaussian_pdf(
            mu=theta_i, 
            sigma=self.equivalent_sigma, 
            thetas=self.thetas
        )