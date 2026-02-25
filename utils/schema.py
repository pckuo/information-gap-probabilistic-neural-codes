# for submission to ICLR 2026


# table for task parameters

import pandas as pd


########################################################################
# single Gaussian task
df_param_SG_dict = {
    '-10_10_10_10': pd.DataFrame({
		'task_id':  [0, 1],
		'mu': 	[-10, 10],
		'sigma':  [10, 10],
        'p_task': [0.5, 0.5]
		})
}


########################################################################
# Cauchy task
df_param_Cauchy_dict = {
    '-10_10_10_10': pd.DataFrame({
		'task_id':  [0, 1],
		'mu': 	[-10, 10],
		'gamma':  [10, 10],
        'p_task': [0.5, 0.5]
		}),
}


########################################################################
# student-t task
df_param_StudentT_dict = {
    '-10_3_10_10_3_10': pd.DataFrame({
		'task_id':  [0, 1],
		'mu': 	[-10, 10],
		'df':  [3, 3],
        'sigma': [10, 10],
        'p_task': [0.5, 0.5]
		}),
}


########################################################################
# table for holding the generated dataset
columns_all = [
	'seed_llh',
    'seed_post',
	'task_id',  # id for task type, starting with zero
	'mu_1',  # parameters for the two gaussian distributions
	'sigma_1',
	'mu_2',
	'sigma_2',
	'classes',  # array of class for each trial
	'orientations',  # array of sampled orientations for each trial
	'stimuli',  # array of filepaths to the generated image stimuli
	'responses_likelihood',  # array (shape of n_unit) of simulated likelihood coding population responses 
	'responses_posterior',  # array (shape of n_unit) of simulated posterior coding population responses 
]
df_sim_data = pd.DataFrame(columns=columns_all)