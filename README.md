# An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes

Code for the paper: "An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes" by Po-Chen Kuo and Edgar Y. Walker. Published in [ICLR 2026](https://openreview.net/forum?id=doxBjZ88H3). 
(arXiv: []())

```
@inproceedings{kuo2026information,
  title={An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes},
  author={Kuo, Po-Chen and Walker, Edgar Y.},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026}
```

This repository provides utilities to compute the information gap, simulate likelihood- and posterior-coding neural populations, and train decoders to compare task designs.



## Repository structure

- [information_gap.ipynb](information_gap.ipynb): end-to-end example for information gap calculation.
- [information_gap/](information_gap/): core information-gap and generative model code.
- [decoder/](decoder/): neural decoder model and training utilities.
- [utils/](utils/): simulation helpers, distributions, and schema definitions.
- [config/](config/): argument configuration for decoder training.

## Setup

This codebase is pure Python. A typical environment uses Python 3.9+ with the following dependencies:

```bash
pip install numpy scipy pandas matplotlib torch jupyter
```

## Usage

### 1) Compute information gap (notebook)

The simplest entry point is the notebook in [information_gap.ipynb](information_gap.ipynb). It defines task priors, a likelihood model, and computes information gap for both likelihood- and posterior-coding populations.

Key components used in the notebook:

- `TaskPrior` and `GaussianLikelihoodModel` from [information_gap/generative_model.py](information_gap/generative_model.py)
- `LikelihoodCodingInformationGapCalculator` and `PosteriorCodingInformationGapCalculator` from [information_gap/information_gap.py](information_gap/information_gap.py)
- Task prior tables in [utils/schema.py](utils/schema.py)

### 2) Programmatic information gap calculation

You can also compute the information gap directly in Python:

```python
import numpy as np

from information_gap.generative_model import TaskPrior, GaussianLikelihoodModel
from information_gap.information_gap import (
	LikelihoodCodingInformationGapCalculator,
	PosteriorCodingInformationGapCalculator,
)
from utils.schema import df_param_SG_dict
import utils.functions as f

# Define stimulus grid
theta_start, theta_end, bins_per_degree = -90, 90, 1
thetas = np.linspace(theta_start, theta_end, int((theta_end - theta_start) * bins_per_degree) + 1)

# Define priors (example: single Gaussian task)
task_name = "-10_10_10_10"
df_task_params = df_param_SG_dict[task_name]
p_task_A, p_task_B = df_task_params["p_task"].iloc[0], df_task_params["p_task"].iloc[1]

prior_A = TaskPrior(
	name="A:SG",
	thetas=thetas,
	distribution=f.get_Gaussian_pdf(df_task_params["mu"].iloc[0], df_task_params["sigma"].iloc[0], thetas),
)
prior_B = TaskPrior(
	name="B:SG",
	thetas=thetas,
	distribution=f.get_Gaussian_pdf(df_task_params["mu"].iloc[1], df_task_params["sigma"].iloc[1], thetas),
)

# Likelihood model
lh_model = GaussianLikelihoodModel(thetas=thetas, equivalent_sigma=15.0)

# Likelihood-coding information gap
lh_calc = LikelihoodCodingInformationGapCalculator(
	lh_model=lh_model,
	prior_A=prior_A,
	prior_B=prior_B,
	p_task_A=p_task_A,
	p_task_B=p_task_B,
)
expected_entropy_lh, info_gap_lh = lh_calc.compute_information_gap()

# Posterior-coding information gap
post_calc = PosteriorCodingInformationGapCalculator(
	lh_model=lh_model,
	prior_A=prior_A,
	prior_B=prior_B,
	p_task_A=p_task_A,
	p_task_B=p_task_B,
)
post_calc._get_pairwise_posterior_kl_divergences()
post_calc._find_pairs_with_identical_posteriors(tolerance=1e-5)
info_gap_post = post_calc.compute_information_gap()

print("Likelihood coding info gap:", info_gap_lh)
print("Posterior coding info gap:", info_gap_post)
```

### 3) Simulate population responses

The simulation utilities in [utils/simulate_population.py](utils/simulate_population.py) support Gaussian tuning curves with Poisson noise and a more complex, bio-realistic gain modulation neural model. Example calls include:

- `simulate_population_responses_likelihood_coding(...)`
- `simulate_population_responses_posterior_coding(...)`

These functions return population responses that can be assembled into datasets for decoder training.

### 4) Train decoders

Decoder training utilities live in [decoder/trainer.py](decoder/trainer.py) and [decoder/model.py](decoder/model.py). Configuration defaults are in [config/args_decoder.py](config/args_decoder.py).

Typical flow:

1. Simulate population responses.
2. Build datasets via `gen_simulation_dataset` in [utils/helpers.py](utils/helpers.py).
3. Initialize `Trainer` and call `train(...)`.

## License

MIT License. See [LICENSE](LICENSE).
