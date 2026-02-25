# for submission to ICLR 2026


import numpy as np
import scipy
from scipy.special import i0



def get_Gaussian_pdf(
    mu, sigma, 
    thetas
):
    """
    return pdf of the Gaussian distribution
    ---
    mu: mean of the Gaussian distribution
    sigma: standard deviation of the Gaussian distribution
    thetas: an array of orientations for computing pdf
    """
    return scipy.stats.norm.pdf(thetas, mu, sigma)


def get_Cauchy_pdf(
    mu, gamma,
    thetas
):
    cauchy_pdf = scipy.stats.cauchy.pdf(
        x=thetas, loc=mu, scale=gamma
    )

    # normalize
    cauchy_pdf /= np.sum(cauchy_pdf)
    if np.abs(cauchy_pdf.sum() - 1.0) >= 0.001:
        raise ValueError(f"the pdf is not properly normalized: {cauchy_pdf.sum()}")

    return cauchy_pdf


def get_StudentT_pdf(
    mu, df, sigma,
    thetas
):
    student_t_pdf = scipy.stats.t.pdf(
        x=thetas, df=df, loc=mu, scale=sigma
    )

    # normalize
    student_t_pdf /= np.sum(student_t_pdf)
    if np.abs(student_t_pdf.sum() - 1.0) >= 0.001:
        raise ValueError(f"the pdf is not properly normalized: {student_t_pdf.sum()}")

    return student_t_pdf


def sample_unbinned_orientations_SingleGaussian(
    mu, sigma, 
    n_trials, 
    random_seed=42
):
    """
    return an array of orientations (size=n_trials)
    sampled from the distribution based on Single Gaussian
    """
    np.random.seed(random_seed)
    sampled_orientations = scipy.stats.norm(loc=mu, scale=sigma).rvs(n_trials)

    return sampled_orientations


def sample_unbinned_orientations_DoubleGaussians(
    mu_1, sigma_1, mu_2, sigma_2, 
    n_trials, 
    random_seed=42
):
    """
    return an array of classes uniformly sampled from the two classes,
    and an array of random orientations (size=n_trials)
    sampled from the distribution based on two Gaussians
    ---
    mu_1, mu_2: means for the two Gaussian distributions
    sigma_1, sigma_2: standard deviations for the two Gaussian distributions
    n_trials: num of trials (samples)
    """
    np.random.seed(random_seed)

    sampled_orientations_all = np.zeros(n_trials)
    # choose one class from the 2 Gaussians for each trial
    classes = np.random.choice([1, 2], size=n_trials)
    # sampling from Gaussian distribution
    orientations_sampled_DG1 = scipy.stats.norm(loc=mu_1, scale=sigma_1).rvs(
        np.sum(classes == 1)
    )
    orientations_sampled_DG2 = scipy.stats.norm(loc=mu_2, scale=sigma_2).rvs(
        np.sum(classes == 2)
    )

    sampled_orientations_all[classes == 1] = orientations_sampled_DG1
    sampled_orientations_all[classes == 2] = orientations_sampled_DG2

    return sampled_orientations_all


def sample_unbinned_orientation_Cauchy(
    mu, gamma, 
    n_trials, 
    random_seed=42
):
    """
    return an array of orientations (size=n_trials)
    sampled from the distribution based on Cauchy
    """
    np.random.seed(random_seed)
    sampled_orientations = scipy.stats.cauchy(loc=mu, scale=gamma).rvs(n_trials)

    return sampled_orientations


def sample_unbinned_orientations_StudentT(
    mu, df, sigma, 
    n_trials, 
    theta_start, theta_end,
    random_seed=42
):
    """
    return an array of orientations (size=n_trials)
    sampled from the distribution based on Student's T
    """
    np.random.seed(random_seed)
    sampled_orientations = scipy.stats.t(df=df, loc=mu, scale=sigma).rvs(n_trials)

    # resample for those outside the range
    while True:
        out_of_bounds = (sampled_orientations < theta_start) | (sampled_orientations > theta_end)
        if not np.any(out_of_bounds):
            break
        sampled_orientations[out_of_bounds] = scipy.stats.t(df=df, loc=mu, scale=sigma).rvs(np.sum(out_of_bounds))

    return sampled_orientations