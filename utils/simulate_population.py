# for submission to ICLR 2026


import numpy as np
import matplotlib.pyplot as plt
import scipy


# set dpi for matplotlib
dpi = 300


######### Gaussian tuning - Poisson noise model #########
def plot_Gaussian_tuning_curves(
    orientations, 
    n_units,
    tuning_amp, tuning_sigma,
    tuning_centers_start, tuning_centers_end, 
    distance_from_edge
):
    '''
    plot Gaussian tuning curves of n_units units that are assumed to tile up
    the orientations homogenously
    ---
    orientations: array of orientations
    n_units: num of units in the population
    tuning_amp: amplitude of the Gaussian tuning curve (assuming homogenous population)
    tuning_sigma: sigma of the Gaussian tuning curve (assuming homogenous population)
    distance_from_edge: distance from the edges of orientation window to be covered
                        by the population of n_units neurons
    '''

    tuning_centers = np.linspace(
        tuning_centers_start+distance_from_edge, 
        tuning_centers_end-distance_from_edge,
        n_units
    )
    # why not -45~45: don't want the edge to be the center of some Gaussian tuning curve

    amps = tuning_amp * np.ones(n_units)
    stds = tuning_sigma * np.ones(n_units)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.set_title(f'Tuning curves: ({tuning_centers_start}, {tuning_centers_end}), n_units={n_units}')
    for unit_id in range(n_units):
        tuning_curve = scipy.stats.norm.pdf(orientations, tuning_centers[unit_id], stds[unit_id])
        tuning_curve = tuning_curve/ tuning_curve.max() * amps[unit_id]
        ax.plot(orientations, tuning_curve,
                label=f'unit {unit_id}')
    ax.set_xlabel('orientation')
    ax.set_ylabel('mean unit activity')
    #ax.legend(ncol=5, bbox_to_anchor=(1, 1))

    fig.tight_layout()
    
    return fig, ax


def population_mean_responses_Gaussian_tuning(
    orientation, 
    n_units,
    tuning_amp, 
    tuning_sigma,
    tuning_centers
):
    '''
    for a given orientation (in degree), 
    return an array of population mean responses (size of n_units)
    based on Gaussian tuning curve models 
    ---
    orientation: int
    n_units: num of units in the population
    tuning_amp: amplitude of the Gaussian tuning curve (assuming homogenous population)
    tuning_sigma: sigma of the Gaussian tuning curve (assuming homogenous population)
    distance_from_edge: distance from the edges of orientation window to be covered
                        by the population of n_units neurons
    '''
    # amps = tuning_amp * np.ones(n_units)
    # stds = tuning_sigma * np.ones(n_units)
    # # expected value of the spike counts for the given orientation
    # population_mean_responses = np.exp(-(orientation-tuning_centers)**2 / 2 / stds**2) * amps

    # construct the Gaussian tuning curve for the given orientation
    population_mean_responses = scipy.stats.norm.pdf(
        orientation, tuning_centers, tuning_sigma
    ) 
    # normalize the Gaussian tuning curve to the tuning amplitude
    population_mean_responses = population_mean_responses / np.max(population_mean_responses)
    population_mean_responses *= tuning_amp
    
    return population_mean_responses


def population_mean_responses_Gamma_gain_modulation(
    population_mean_responses,
    random_seed,
    gamma_variance=0.5,  # estimated from Goris et al., 2014
):
    '''
    return an array (size of n_units) of mean responses modulated by gain sampled from Gamma distribution
    '''
    np.random.seed(random_seed)
    # sample from Gamma distribution
    gain_gamma_samples = np.random.gamma(
        shape=1/gamma_variance, 
        scale=gamma_variance, 
        size=population_mean_responses.shape
    )
    # modulate the mean responses
    population_mean_responses_gain_modulated = population_mean_responses * gain_gamma_samples
    return population_mean_responses_gain_modulated



def population_responses_poisson_noise(
    population_mean_responses,
    random_seed,
    gain=1
):
    '''
    return an array (size of n_units) of spike counts sampled from 
    Poisson distributions based on the mean responses derived from Gaussian tuning curves
    ---
    population_mean_responses: array (size of n_units) of mean responses for a given orientation
                               derived from the Gaussian tuning curve models
    gain: int, gain to amplify the mean responses
    '''
    np.random.seed(random_seed)
    mus = (population_mean_responses * gain).T  # make it into n_trials x n_units
    population_responses = np.random.poisson(mus)
    
    return population_responses


def simulate_population_responses_likelihood_coding(
    stim_orientations,
    n_units,
    tuning_amp,
    tuning_sigma,
    tuning_centers,
    random_seed,
    poisson_noise=True,
    poisson_gain=1,
    gamma_gain_modulation=False
):
    '''
    return an array (size: n_trials x n_units) of simulated likelihood coding
    population responses based on the input orientations array
    ---
    orientations: an array (size of n_trials) of orientations for each trial
    '''
    np.random.seed(random_seed)

    population_responses_all = []
    for orientation in stim_orientations:
        random_seed_trial = np.random.randint(0, 10000, size=1)

        mean_responses_Gaussian = population_mean_responses_Gaussian_tuning(
            orientation,
            n_units,
            tuning_amp, 
            tuning_sigma,
            tuning_centers
        )

        if gamma_gain_modulation:
            mean_responses_Gaussian = population_mean_responses_Gamma_gain_modulation(
                mean_responses_Gaussian,
                random_seed_trial
            )

        if poisson_noise:
            spike_counts_poisson = population_responses_poisson_noise(
                mean_responses_Gaussian, 
                random_seed_trial,
                poisson_gain
            )
        else:
            spike_counts_poisson = mean_responses_Gaussian

        population_responses_all.append(spike_counts_poisson)
    
    population_responses_all = np.array(population_responses_all)  # (n_trials, n_units)
    
    return population_responses_all


########################################################
# POSTERIOR CODING POPULATION
def population_mean_responses_prior_modulation(
    mean_responses_likelihood, 
    prior_task,
    tuning_amp
    ):
    '''
    for a given orientation (in degree), 
    return an array of population mean responses (size of n_units)
    that are modulated by task prior 
    ---
    orientation: int
    n_units: num of units in the population
    tuning_amp: amplitude of the Gaussian tuning curve (assuming homogenous population)
    distance_from_edge: distance from the edges of orientation window to be covered
                        by the population of n_units neurons
    '''
    # expected value of the spike counts for the given orientation with prior modulation
    mean_responses_prior_modulated = mean_responses_likelihood * prior_task
    # print(f'mean_responses_prior_modulated: {mean_responses_prior_modulated}')
    
    # TODO: should we normalize the mean responses?

    mean_responses_prior_modulated *= (tuning_amp/ np.max(mean_responses_prior_modulated))
    # print(f'mean_responses_prior_modulated: {mean_responses_prior_modulated}')
    
    return mean_responses_prior_modulated


def simulate_population_responses_posterior_coding(
    stim_orientations,
    n_units,
    tuning_amp, tuning_sigma,
    tuning_centers,
    task_prior,
    random_seed,
    poisson_noise=True,
    poisson_gain=1,
    gamma_gain_modulation=False
):
    '''
    return an array (size: n_trials x n_units) of simulated posterior coding
    population responses based on stim_orientations and prior
    ---
    orientations: an array (size: n_trials) of orientations for each trial
    '''
    np.random.seed(random_seed)

    population_responses_all = []
    for orientation in stim_orientations:
        random_seed_trial = np.random.randint(0, 10000, size=1)

        mean_responses_Gaussian = population_mean_responses_Gaussian_tuning(
            orientation,
            n_units,
            tuning_amp, 
            tuning_sigma,
            tuning_centers
        )
        mean_responses_prior_modulated = population_mean_responses_prior_modulation(
            mean_responses_Gaussian,
            task_prior,
            tuning_amp
        )

        if gamma_gain_modulation:
            mean_responses_prior_modulated = population_mean_responses_Gamma_gain_modulation(
                mean_responses_prior_modulated,
                random_seed_trial
            )

        if poisson_noise:
            spike_counts_poisson = population_responses_poisson_noise(
                mean_responses_prior_modulated, 
                random_seed_trial, 
                poisson_gain
            )
        else:
            spike_counts_poisson = mean_responses_prior_modulated
            
        population_responses_all.append(spike_counts_poisson)
        # plt.plot(mean_responses_prior_modulated)
    
    population_responses_all = np.array(population_responses_all)  # (n_trials, n_units)
    
    return population_responses_all