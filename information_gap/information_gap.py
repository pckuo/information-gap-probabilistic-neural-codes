# for submission to ICLR 2026


from typing import Tuple, List, Dict

import numpy as np
# from scipy.special import rel_entr
from scipy.stats import entropy
import matplotlib.pyplot as plt

from information_gap.generative_model import TaskPrior, LikelihoodModel


dpi = 300

def get_kl_divergence( 
        p: np.ndarray, 
        q: np.ndarray
    ) -> float:
    """Compute KL divergence D_KL(p || q)"""
    # Use scipy's entropy function which computes KL divergence
    # add small epsilon to avoid log(0)
    p = p + 1e-10  # Avoid log(0)
    q = q + 1e-10  # Avoid log(0)
    # note that scipy's entropy function handles normalization internally
    # so we don't need to normalize p and q explicitly
    return entropy(p, q)  # KL divergence D_KL(p || q)


class BaseInformationGapCalculator:
    """Base class for information gap calculators"""
    
    def __init__(
        self, 
        lh_model: LikelihoodModel,
        prior_A: TaskPrior,
        prior_B: TaskPrior,
        p_task_A: float = 0.5,
        p_task_B: float = 0.5
    ):
        self.lh_model = lh_model
        self.prior_A = prior_A
        self.prior_B = prior_B
        self.p_task_A = p_task_A
        self.p_task_B = p_task_B

    def compute_information_gap(
        self, 
    ) -> float:
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError
    

class LikelihoodCodingInformationGapCalculator(BaseInformationGapCalculator):
    """Calculates information gap for likelihood coding populations"""

    def _get_unnormalized_context_frequency_xi(
        self,
        x_i
    ) -> Tuple[float, float]:
        """
        Compute context frequency for a given x_i.
        """
        # Get the likelihood for x_i
        likelihood_xi = self.lh_model.get_likelihood_function(x_i)
        
        # Compute context frequency for A and B
        unnormalized_freq_A_xi = self.p_task_A * np.sum(likelihood_xi * self.prior_A.distribution)
        unnormalized_freq_B_xi = self.p_task_B * np.sum(likelihood_xi * self.prior_B.distribution)

        return unnormalized_freq_A_xi, unnormalized_freq_B_xi

    def _get_surrogate_prior_xi(
        self,
        x_i
    ) -> np.ndarray:
        """Compute the surrogate prior for the given x_i."""
        
        # Compute the surrogate prior as a mixture of the two priors
        # Get the unnormalized context frequency for x_i
        unnormalized_context_freq_A_xi, unnormalized_context_freq_B_xi = self._get_unnormalized_context_frequency_xi(x_i)
        
        # Compute the surrogate prior as a weighted average of the two priors
        surrogate_prior_xi = (
            self.p_task_A * self.prior_A.distribution + 
            self.p_task_B * self.prior_B.distribution
        )

        if np.abs(surrogate_prior_xi.sum() - 1.0) >= 0.001:
            raise ValueError(f"the surrogate prior is not properly normalized: {surrogate_prior_xi.sum()}")
        
        return surrogate_prior_xi
    
    def _get_information_gap_per_xi(
        self, 
        x_i,
        plotting=False
    ) -> Dict:
        """
        For a given x_i, compute the 
        1. unnormalized context frequency 
        2. posterior entropy 
        3. surrogate prior & posterior
        4. information gap 
        """
        likelihood_xi = self.lh_model.get_likelihood_function(x_i)

        # context frequency for A and B
        unnormalized_context_freq_A_xi, unnormalized_context_freq_B_xi = self._get_unnormalized_context_frequency_xi(x_i)
        

        # output of a perfect likelihood decoder
        # true posteriors under each task
        posterior_A_xi = self.lh_model.get_posterior(x_i, self.prior_A)
        posterior_B_xi = self.lh_model.get_posterior(x_i, self.prior_B)

        posterior_entropy_A_xi = entropy(posterior_A_xi + 1e-10)  # Avoid log(0)
        posterior_entropy_B_xi = entropy(posterior_B_xi + 1e-10)  # Avoid log(0)
        
        
        # output of the best possible posterior decoder
        # mixture of true posteriors
        surrogate_prior_xi = self._get_surrogate_prior_xi(x_i)
        # Surrogate posterior
        surrogate_posterior_xi = likelihood_xi * surrogate_prior_xi
        surrogate_posterior_xi = surrogate_posterior_xi / np.sum(surrogate_posterior_xi)  # Normalize

        # Compute KL divergence for information gap
        info_gap_A_xi = get_kl_divergence(posterior_A_xi, surrogate_posterior_xi)
        info_gap_B_xi = get_kl_divergence(posterior_B_xi, surrogate_posterior_xi)


        if plotting:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f"Likelihood coding information gap for x={x_i}")

            # plot the likelihoods
            ax.plot(
                self.lh_model.thetas, 
                likelihood_xi, 
                'k', 
                label='Likelihood'
            )
            # plot the true priors and posteriors
            ax.plot(
                self.lh_model.thetas, 
                self.prior_A.distribution, 
                'b--', alpha=0.5, 
                label='Prior A'
            )
            ax.plot(
                self.lh_model.thetas, 
                self.prior_B.distribution, 
                'r--', alpha=0.5, 
                label='Prior B'
            )
            ax.plot(
                self.lh_model.thetas, 
                posterior_A_xi, 
                'b', 
                label='Posterior A'
            )
            ax.plot(
                self.lh_model.thetas, 
                posterior_B_xi, 
                'r', 
                label='Posterior B'
            )
            # plot the surrogate prior and posterior
            ax.plot(
                self.lh_model.thetas, 
                surrogate_prior_xi, 
                'g--', alpha=0.5, 
                label='Surrogate Prior'
            )
            ax.plot(
                self.lh_model.thetas, 
                surrogate_posterior_xi, 
                'g', 
                label='Surrogate Posterior'
            )

            # add text for posterior entropy and information gap
            ax.text(
                0.01, 0.9, 
                f"Posterior Entropy A: {posterior_entropy_A_xi:.2f}\nPosterior Entropy B: {posterior_entropy_B_xi:.2f}\n"
                f"Info_gap A: {info_gap_A_xi:.2f}\nInfo_gap B: {info_gap_B_xi:.2f}", 
                transform=plt.gca().transAxes, 
                fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.5)
            )
            
            ax.set_xlabel("Theta")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid()

            fig.tight_layout()
        
        return {
            'unnormalized_context_freq_A_xi': unnormalized_context_freq_A_xi,
            'unnormalized_context_freq_B_xi': unnormalized_context_freq_B_xi,
            'posterior_entropy_A_xi': posterior_entropy_A_xi,
            'posterior_entropy_B_xi': posterior_entropy_B_xi,
            'surrogate_prior_xi': surrogate_prior_xi,
            'surrogate_posterior_xi': surrogate_posterior_xi,
            'info_gap_A_xi': info_gap_A_xi,
            'info_gap_B_xi': info_gap_B_xi
        }

    def compute_information_gap(
        self, 
    ) -> float:
        """
        Compute the posterior entropy and information gap 
        for a likelihood coding population

        When a likelihood coding population is decoded with:
        - Likelihood decoder: perfect decoding
        - Posterior decoder: outputs mixture of posteriors
        """
        expected_posterior_entropy = 0.0  # perfect likelihood decoder cross entropy
        info_gap = 0.0  # information gap between posterior and surrogate posterior
        
        for x_i in self.lh_model.thetas:
            results = self._get_information_gap_per_xi(
                x_i, 
                plotting=False
            )
            
            # posterior entropy: add weighted entropy contributions
            expected_posterior_entropy += (
                results['unnormalized_context_freq_A_xi'] * results['posterior_entropy_A_xi'] +
                results['unnormalized_context_freq_B_xi'] * results['posterior_entropy_B_xi']
            )

            # Information gap contributions
            info_gap += (
                results['unnormalized_context_freq_A_xi'] * results['info_gap_A_xi'] +
                results['unnormalized_context_freq_B_xi'] * results['info_gap_B_xi']
            )

        return expected_posterior_entropy, info_gap
    

class PosteriorCodingInformationGapCalculator(BaseInformationGapCalculator):
    """Calculates information gap for posterior coding populations"""

    def _get_unnormalized_context_frequency_xj_xk(
        self,
        x_j,
        x_k
    ) -> Tuple[float, float]:
        """
        Compute context frequency for a given pari of x_j and x_k.
        """
        # Get the likelihood for x_j and x_k
        likelihood_xj = self.lh_model.get_likelihood_function(x_j)
        likelihood_xk = self.lh_model.get_likelihood_function(x_k)

        # Compute context frequency for A and B
        unnormalized_freq_A_xj = self.p_task_A * np.sum(likelihood_xj * self.prior_A.distribution)
        unnormalized_freq_B_xk = self.p_task_B * np.sum(likelihood_xk * self.prior_B.distribution)

        return unnormalized_freq_A_xj, unnormalized_freq_B_xk
    
    def _get_pairwise_posterior_kl_divergences(
        self,
        plotting: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        pairs = []
        pairwise_posterior_kl_divergences = []

        for x_j in self.lh_model.thetas:
            for x_k in self.lh_model.thetas:
                # Compute KL divergence between posteriors
                posterior_A_xj = self.lh_model.get_posterior(x_j, self.prior_A)
                posterior_B_xk = self.lh_model.get_posterior(x_k, self.prior_B)
                posterior_kl_divergence = get_kl_divergence(posterior_A_xj, posterior_B_xk)
                
                # Store the pair and its KL divergence
                pairs.append((x_j, x_k))
                pairwise_posterior_kl_divergences.append(posterior_kl_divergence)

        # make nump arrays
        pairs = np.array(pairs)
        pairwise_posterior_kl_divergences = np.array(pairwise_posterior_kl_divergences)

        # save as attributes
        self.pairs = pairs
        self.pairwise_posterior_kl_divergences = pairwise_posterior_kl_divergences


        if plotting:
            # make 4 subplots for histograms of KL divergences
            # one for all kl values
            fig, axs = plt.subplots(2, 2, figsize=(16, 8), dpi=dpi)
            for row_idx in range(2):
                for col_idx in range(2):
                    ax = axs[row_idx, col_idx]
                    ax.spines[['top', 'right']].set_visible(False)

                    if row_idx == 0 and col_idx == 0:
                        # all kl values
                        ax_title = 'Histogram of Pairwise KL Divergences'
                        data_for_hist = pairwise_posterior_kl_divergences
                        bins = np.linspace(0, 20, 100)
                    elif row_idx == 0 and col_idx == 1:
                        # kl values between 0 and 1
                        ax_title = 'Histogram of Pairwise KL Divergences (0 to 1)'
                        data_for_hist = pairwise_posterior_kl_divergences[(pairwise_posterior_kl_divergences <= 1)]
                        bins = np.linspace(0, 1, 100)
                    elif row_idx == 1 and col_idx == 0:
                        # kl values between 0 and 0.1
                        ax_title = 'Histogram of Pairwise KL Divergences (0 to 0.1)'
                        data_for_hist = pairwise_posterior_kl_divergences[(pairwise_posterior_kl_divergences <= 0.1)]
                        bins = np.linspace(0, 0.1, 100)
                    elif row_idx == 1 and col_idx == 1:
                        # kl values between 0 and 0.001
                        ax_title = 'Histogram of Pairwise KL Divergences (0 to 0.001)'
                        data_for_hist = pairwise_posterior_kl_divergences[(pairwise_posterior_kl_divergences <= 0.001)]
                        bins = np.linspace(0, 0.001, 100)

                    ax.hist(
                        data_for_hist,
                        bins=bins,
                        color='gray', alpha=0.7
                    )
                    ax.set_title(ax_title)
                    ax.set_xlabel('KL Divergence')
                    ax.set_ylabel('Frequency')
                    ax.grid()

            fig.tight_layout()

        return pairs, pairwise_posterior_kl_divergences

    def _find_pairs_with_identical_posteriors(
        self,
        tolerance: float = 1e-5
    ) -> List[Tuple[int, int]]:
        """Find pairs (xj, xk) where p^A(θ|x_j) = p^B(θ|x_k)"""
        
        if not hasattr(self, 'pairwise_posterior_kl_divergences'):
            # if pairwise_kl_divergences not computed, compute it
            self._get_pairwise_posterior_kl_divergences(plotting=False)

        # find pairs with KL divergence below the tolerance
        pairs_with_identical_posteriors = self.pairs[
            self.pairwise_posterior_kl_divergences <= tolerance
        ]
        print(f"Using dynamic tolerance for identical posteriors: {tolerance:.5f}")
        print(f"Found {len(pairs_with_identical_posteriors)} pairs with identical posteriors")

        # refine pair selection by selecting the smallest KL divergence per duplicate first theta
        unique_first_thetas = np.unique(pairs_with_identical_posteriors[:, 0])
        refined_pairs = []
        for unique_theta in unique_first_thetas:
            # find all pairs with this theta
            matching_pairs = pairs_with_identical_posteriors[pairs_with_identical_posteriors[:, 0] == unique_theta]
            if matching_pairs.size > 0:
                # select the pair with the smallest KL divergence
                matching_pair_kls = []
                for matching_pair in matching_pairs:
                    matching_pair_id = np.where((self.pairs == matching_pair).all(axis=1))[0][0]
                    matching_pair_kls.append(self.pairwise_posterior_kl_divergences[matching_pair_id])
                min_kl_pair = matching_pairs[np.argmin(np.array(matching_pair_kls))]
                refined_pairs.append(min_kl_pair)
            else:
                refined_pairs.append(matching_pairs)  # in case no duplicate matching pairs found, just append the first one

        print(f"Refined pairs with identical posteriors: {len(refined_pairs)}")

        self.pairs_with_identical_posteriors = refined_pairs

        return refined_pairs
    

    def _plot_pairs_with_identical_posteriors(
        self,
        n_pairs_to_plot: int = 1
    ) -> None:
        """Plot pairs (xj, xk) where p^A(θ|x_j) = p^B(θ|x_k)"""
        if not hasattr(self, 'pairs_with_identical_posteriors'):
            raise ValueError("pairs_with_identical_posteriors not computed. Call _find_pairs_with_identical_posteriors() first.")
        
        # randomly select pairs to plot
        if n_pairs_to_plot > len(self.pairs_with_identical_posteriors):
            raise ValueError(f"n_pairs_to_plot ({n_pairs_to_plot}) exceeds available constraint pairs ({len(self.pairs_with_identical_posteriors)})")
        
        # np.random.seed(42)  # for reproducibility
        np.random.shuffle(self.pairs_with_identical_posteriors)
        # Select the first n_pairs_to_plot pairs
        pairs_to_plot = self.pairs_with_identical_posteriors[:n_pairs_to_plot]
        
        # plot each pair as a separate figure
        for (x_j, x_k) in pairs_to_plot:
            # compute information gap for this pair
            results = self._get_information_gap_per_identical_posterior_pair(
                x_j, 
                x_k
            )             
        
            fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'Pair with identical posteriors: (Task A xj={x_j}, Task B xk={x_k})')

            # plot the posteriors
            ax.plot(
                self.lh_model.thetas, 
                results['posterior_A_xj'], 
                'k', 
                label=f'Posterior A (xj={x_j})')
            ax.plot(
                self.lh_model.thetas, 
                results['posterior_B_xk'], 
                'darkslategray', 
                label=f'Posterior B (xk={x_k})')
            # plot the likelihoods
            likelihood_j = self.lh_model.get_likelihood_function(x_j)
            likelihood_k = self.lh_model.get_likelihood_function(x_k)
            ax.plot(
                self.lh_model.thetas, 
                likelihood_j, 
                'b', 
                label=f'Likelihood (xj={x_j})')
            ax.plot(
                self.lh_model.thetas, 
                likelihood_k, 
                'r', 
                label=f'Likelihood (xk={x_k})')
            # plot the priors
            ax.plot(
                self.lh_model.thetas, 
                self.prior_A.distribution, 
                'b--', alpha=0.5, 
                label='Prior A')
            ax.plot(
                self.lh_model.thetas, 
                self.prior_B.distribution, 
                'r--', alpha=0.5, 
                label='Prior B')
            
            # plot the surrogate likelihood
            ax.plot(
                self.lh_model.thetas, 
                results['surrogate_likelihood'], 
                'g', 
                label='Surrogate Likelihood'
            )
            # plot the surrogate posteriors
            ax.plot(
                self.lh_model.thetas, 
                results['surrogate_posterior_A_xj'], 
                'purple', 
                label=f'Surrogate Posterior A')
            ax.plot(
                self.lh_model.thetas, 
                results['surrogate_posterior_B_xk'], 
                'peru', 
                label=f'Surrogate Posterior B')
            
            # add text for information gap
            ax.text(
                0.01, 0.9, 
                f"Info_gap A: {results['info_gap_A_xj']:.2f}\nInfo_gap B: {results['info_gap_B_xk']:.2f}", 
                transform=plt.gca().transAxes, 
                fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.5)
            )
            
            ax.set_xlabel('Theta')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
            ax.grid()

    def _solve_best_likelihood_fixed_point_iteration(
        self,
        weighted_posterior_sum_Aj_Bk,
        normalized_context_freq_A_xj,
        normalized_context_freq_B_xk,
        iteration_tolerance: float = 1e-5,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the fixed point iteration to find the best likelihood decoder output.
        """
        # Initialize the surrogate_likelihood
        surrogate_likelihood = np.ones_like(self.lh_model.thetas)/ len(self.lh_model.thetas)  # Uniform distribution

        # Fixed point iteration
        for iter_idx in range(max_iterations):
            normalization_factor_Aj = np.sum(surrogate_likelihood * self.prior_A.distribution)
            normalization_factor_Bk = np.sum(surrogate_likelihood * self.prior_B.distribution)

            weighted_prior_sum_Aj_Bk = (
                normalized_context_freq_A_xj / normalization_factor_Aj * self.prior_A.distribution + 
                normalized_context_freq_B_xk / normalization_factor_Bk * self.prior_B.distribution
            )
            
            # update the surrogate likelihood
            updated_surrogate_likelihood = (
                weighted_posterior_sum_Aj_Bk / 
                (weighted_prior_sum_Aj_Bk + 1e-10)  # Avoid division by zero
            )
            updated_surrogate_likelihood /= np.sum(updated_surrogate_likelihood)  # Normalize

            # Check for convergence
            # If the maximum change in surrogate likelihood is less than the tolerance, we can stop
            if np.max(np.abs(updated_surrogate_likelihood - surrogate_likelihood)) < iteration_tolerance:
                # print(f"Converged after {_} iterations")
                break
            else:
                surrogate_likelihood = updated_surrogate_likelihood
                # if exceeds max iterations, print a warning
                if iter_idx == max_iterations-1:
                    print(f"Warning: Max iterations reached without convergence: max_change={np.max(np.abs(updated_surrogate_likelihood - surrogate_likelihood)):.6f}")
            
            # return the surrogate likelihood
        return surrogate_likelihood

    def _get_information_gap_per_identical_posterior_pair(
        self, 
        x_j, 
        x_k
    ) -> Dict:
        """
        For a given identical posterior pairs of (x_j, x_k), compute the 
        1. unnormalized context frequency 
        2. posterior entropy
        3. weighted posterior sum
        4. weighted prior sum (use fixed point iteration)
        5. surrogate likelihood (decoder output)
        6. surrogate posterior
        7. information gap for
        """
        likelihood_xj = self.lh_model.get_likelihood_function(x_j)
        likelihood_xk = self.lh_model.get_likelihood_function(x_k)

        # get context frequency
        unnormalized_context_freq_A_xj, unnormalized_context_freq_B_xk = self._get_unnormalized_context_frequency_xj_xk(
            x_j, 
            x_k
        )
        normalized_context_freq_A_xj = unnormalized_context_freq_A_xj / (unnormalized_context_freq_A_xj + unnormalized_context_freq_B_xk + 1e-10)  # Avoid division by zero
        normalized_context_freq_B_xk = unnormalized_context_freq_B_xk / (unnormalized_context_freq_A_xj + unnormalized_context_freq_B_xk + 1e-10)


        # output of the best possible likelihood decoders
        # get weighted surrogate posterior sum
        posterior_A_xj = self.lh_model.get_posterior(x_j, self.prior_A)
        posterior_B_xk = self.lh_model.get_posterior(x_k, self.prior_B)
        weighted_posterior_A_xj = normalized_context_freq_A_xj * posterior_A_xj
        weighted_posterior_B_xk = normalized_context_freq_B_xk * posterior_B_xk
        weighted_posterior_sum_Aj_Bk = (
            weighted_posterior_A_xj + 
            weighted_posterior_B_xk
        )

        # Solve the fixed point iteration to find the best likelihood decoder output
        surrogate_likelihood = self._solve_best_likelihood_fixed_point_iteration(
            weighted_posterior_sum_Aj_Bk,
            normalized_context_freq_A_xj,
            normalized_context_freq_B_xk,
            iteration_tolerance=1e-5
        )

        # Surrogate posterior
        surrogate_posterior_A_xj = surrogate_likelihood * self.prior_A.distribution
        surrogate_posterior_A_xj = surrogate_posterior_A_xj / np.sum(surrogate_posterior_A_xj)

        surrogate_posterior_B_xk = surrogate_likelihood * self.prior_B.distribution
        surrogate_posterior_B_xk = surrogate_posterior_B_xk / np.sum(surrogate_posterior_B_xk)

        # Compute KL divergence for information gap
        info_gap_A = get_kl_divergence(posterior_A_xj, surrogate_posterior_A_xj)
        info_gap_B = get_kl_divergence(posterior_B_xk, surrogate_posterior_B_xk)

        return {
            'unnormalized_context_freq_A_xj': unnormalized_context_freq_A_xj,
            'unnormalized_context_freq_B_xk': unnormalized_context_freq_B_xk,
            'normalized_context_freq_A_xj': normalized_context_freq_A_xj,
            'normalized_context_freq_B_xk': normalized_context_freq_B_xk,
            'posterior_A_xj': posterior_A_xj,
            'posterior_B_xk': posterior_B_xk,
            'surrogate_likelihood': surrogate_likelihood,
            'surrogate_posterior_A_xj': surrogate_posterior_A_xj,
            'surrogate_posterior_B_xk': surrogate_posterior_B_xk,
            'info_gap_A_xj': info_gap_A,
            'info_gap_B_xk': info_gap_B
        }

    def compute_information_gap(
        self
    ) -> float:
        """
        Compute information gap for posterior coding population
        
        When posterior coding population is decoded with:
        - Posterior decoder: perfect decoding
        - Likelihood decoder: complex mapping due to constraints
        """
        info_gap = 0.0
        
        # Find pairs (j, k) where p^A(θ_j|x) = p^B(θ_k|x)
        if not hasattr(self, 'pairs_with_identical_posteriors'):
            raise ValueError("pairs_with_identical_posteriors not computed. Call _find_pairs_with_identical_posteriors() first.")

        for (x_j, x_k) in self.pairs_with_identical_posteriors:
            # get information gap for this pair
            results = self._get_information_gap_per_identical_posterior_pair(
                x_j,
                x_k
            )

            # Information gap contributions
            info_gap += (
                results['unnormalized_context_freq_A_xj'] * results['info_gap_A_xj'] +
                results['unnormalized_context_freq_B_xk'] * results['info_gap_B_xk']
            )
            
        return info_gap