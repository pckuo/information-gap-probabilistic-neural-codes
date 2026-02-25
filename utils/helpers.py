# for submission to ICLR 2026


from distutils.util import strtobool

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


# generate dataset
def gen_simulation_dataset(
    df_sim_data,
    theta_start,
    theta_end,
    n_units,
    n_units_max=500
):
    '''
    return tensor datasets for lh and post coding population
    '''
    true_binned_thetas = []
    responses_lh_w_task_ids_w_task_priors = []
    responses_post_w_task_ids_w_task_priors = []

    # select units
    ## even spacing
    selected_unit_ids = np.linspace(0, n_units_max-1, n_units, dtype=int)
    # ## random sampling
    # selected_unit_ids = np.sort(random.sample(range(n_units_max), n_units))


    for row_id, row in df_sim_data.iterrows():
        # true binned thetas
        true_unbinned_thetas_session = row['true_sampled_unbinned_thetas']
        # true_binned_thetas_session = np.floor(true_unbinned_thetas_session)
        # use rint for rounding to nearest integer
        true_binned_thetas_session = np.rint(true_unbinned_thetas_session)
        shifted_true_binned_thetas_session = true_binned_thetas_session - theta_start

        # drop out-of-range binned theta trials
        to_drop_index = np.where(
            (shifted_true_binned_thetas_session < 0) | 
            (shifted_true_binned_thetas_session > (theta_end - theta_start))
        )
        # print(f'to_drop_index: {to_drop_index}')
        selected_true_binned_thetas_session = np.delete(
            shifted_true_binned_thetas_session, to_drop_index
        )

        # task_id
        task_id_session = row['task_id']
        task_ids_session = np.ones((true_unbinned_thetas_session.shape[0], 1)) * task_id_session
        # shape: (n_trials, 1)

        # task_prior_for_decoder
        task_prior_for_decoder_session = row['task_prior_for_decoder']
        task_priors_for_decoder_session = np.tile(
            task_prior_for_decoder_session, 
            (true_unbinned_thetas_session.shape[0], 1)
        )  # shape: (n_trials, n_bins)

        # lh coding population
        responses_lh_session = row['responses_likelihood'][:, selected_unit_ids]  # (n_trials, n_units)
        responses_lh_w_task_ids_w_task_priors_session = np.hstack(
            (responses_lh_session, task_ids_session, task_priors_for_decoder_session)
        )  # (n_trials, n_units + 1 + n_bins)
        selected_responses_lh_w_task_ids_w_task_priors_session = np.delete(
            responses_lh_w_task_ids_w_task_priors_session, to_drop_index, 0
        )

        # post coding population
        responses_post_session = row['responses_posterior'][:, selected_unit_ids]  # (n_trials, n_units)
        responses_post_w_task_ids_w_task_priors_session = np.hstack(
            (responses_post_session, task_ids_session, task_priors_for_decoder_session)
        )  # (n_trials, n_units + 1 + n_bins)
        selected_responses_post_w_task_ids_w_task_priors_session = np.delete(
            responses_post_w_task_ids_w_task_priors_session, to_drop_index, 0
        )
        
        # append data
        true_binned_thetas.append(selected_true_binned_thetas_session)
        responses_lh_w_task_ids_w_task_priors.append(selected_responses_lh_w_task_ids_w_task_priors_session)
        responses_post_w_task_ids_w_task_priors.append(selected_responses_post_w_task_ids_w_task_priors_session)

    # concatenate all sessions
    true_binned_thetas = np.concatenate(true_binned_thetas)
    responses_lh_w_task_ids_w_task_priors = np.concatenate(responses_lh_w_task_ids_w_task_priors)
    responses_post_w_task_ids_w_task_priors = np.concatenate(responses_post_w_task_ids_w_task_priors)

    print(f'true_binned_thetas: {true_binned_thetas.shape}')
    print(f'simulated likelihood responses with task_ids with task_priors: {responses_lh_w_task_ids_w_task_priors.shape}')
    print(f'simulated posterior responses with task_ids with task_priors: {responses_post_w_task_ids_w_task_priors.shape}')

    true_binned_thetas = torch.from_numpy(true_binned_thetas.astype(int))
    responses_lh_w_task_ids_w_task_priors = torch.from_numpy(responses_lh_w_task_ids_w_task_priors)
    responses_post_w_task_ids_w_task_priors = torch.from_numpy(responses_post_w_task_ids_w_task_priors)

    lh_dataset = TensorDataset(responses_lh_w_task_ids_w_task_priors, true_binned_thetas)
    post_dataset = TensorDataset(responses_post_w_task_ids_w_task_priors, true_binned_thetas)

    return lh_dataset, post_dataset


def evaluate_decoder(
    args, 
    decoder, 
    flex_log_prior_diff,  # for flex decoder
    dataset,
    delta=1
):
    decoder.eval()

    # prepare prior
    if args.decoder_type == 'flex':
        fixed_log_prior_reference = torch.zeros(args.output_dim, requires_grad=False)
        flex_log_prior_diff = torch.from_numpy(flex_log_prior_diff)
        log_prior_flex_decoder = torch.stack((fixed_log_prior_reference, flex_log_prior_diff))

    ce_loss = 0
    mse_loss_map = 0
    mse_loss_mean = 0
    correct_count = 0
    pred_log_lhs, associated_log_priors, pred_log_posts = [], [], []
    
    remove_count = 0
    with torch.no_grad():
        for test_id in range(len(dataset)):
            x_, t_ = dataset[test_id]

            t = t_.to(args.device)

            x_ = x_.type(torch.FloatTensor).to(args.device)  # (n_units+1)
            x = x_[:args.input_dim]  # (n_units)
            
            if args.decoder_type == 'lh':
                log_prior = torch.log(x_[args.input_dim+1:].to(args.device))  # (n_bins)
            elif args.decoder_type == 'flex':
                task_id = x_[args.input_dim].long()
                log_prior = log_prior_flex_decoder[task_id]

            if args.decoder_type == 'post':
                # model output  -> log posterior
                y = decoder(x)  
                log_post = y
            elif args.decoder_type in ['lh', 'flex']:
                # model output  -> log likelihood
                y = decoder(x)
                log_post = y + log_prior.to(args.device)  # adding log prior -> log posterior
                pred_log_lhs.append(y.data.cpu().numpy())
                associated_log_priors.append(log_prior.data.cpu().numpy())
            pred_log_posts.append(log_post.data.cpu().numpy())

            # shift to avoid numerical overflow
            val, _ = log_post.max(0, keepdim=True)
            log_post = log_post - val
            
            # -- compute loss --
            # cross entropy loss
            ce_loss_batch = F.cross_entropy(log_post, t)
            ce_loss += ce_loss_batch.data.cpu().numpy()
        
            # mse loss: max a posteriori
            _, loc = torch.max(log_post, dim=0)
            # print(f' target - MAP: {t} - {loc}')
            if t.double() == loc.double():
                correct_count += 1
            mse_loss_map_batch = (t.double() - loc.double()).pow(2).mean().sqrt() * delta
            mse_loss_map += mse_loss_map_batch.data.cpu().numpy()

            # mse loss: mean
            post = torch.exp(log_post)
            norm = torch.norm(post)
            normalized_post = (post / norm).data.cpu().numpy().reshape(args.output_dim)
            bins = np.arange(0, args.output_dim)
            try: 
                expectation_val = np.average(bins, axis=0, weights=normalized_post)
                mse_loss_mean_batch = np.sqrt(np.mean((expectation_val - t.data.cpu().numpy())**2))
                if np.isnan(mse_loss_mean_batch):
                    print('nan error! mse_loss_mean_batch: {}, target: {}, mean from post: {}, max a priori: {}'.format(
                        mse_loss_mean_batch, t.data.cpu().numpy(), expectation_val, loc.double().data.cpu().numpy()))
                    print(' nan error! orginal log_post: {}'.format(log_post))
            except ZeroDivisionError:
                expectation_val = -1
                mse_loss_mean_batch = 0
                remove_count += 1
            if not np.isnan(mse_loss_mean_batch):
                mse_loss_mean += mse_loss_mean_batch
    
    if remove_count > 0:
        print(f' remove count: {remove_count}')
    
    avg_ce_loss = ce_loss/ len(dataset)
    avg_mse_loss_map = mse_loss_map/ len(dataset)
    avg_mse_loss_mean = mse_loss_mean/ (len(dataset) - remove_count)
    correct_rate = float(correct_count)/ len(dataset)

    print('test dataset size: {}'.format(len(dataset)))
    print(f'total mse_loss_mean: {mse_loss_mean}')
    print(f'avg_ce_loss: {avg_ce_loss}')
    print(f'avg_mse_loss_map: {avg_mse_loss_map}')
    print(f'avg_mse_loss_mean: {avg_mse_loss_mean}')
    print(f'correct rate: {correct_rate}')
    
    if args.decoder_type == 'post':
        pred_log_lhs = None
        associated_log_priors = None
    elif args.decoder_type == 'lh':
        pred_log_lhs = np.concatenate(pred_log_lhs, axis=0)
        associated_log_priors = None
    elif args.decoder_type == 'flex':
        pred_log_lhs = np.concatenate(pred_log_lhs, axis=0)
        associated_log_priors = np.concatenate(associated_log_priors, axis=0)
    else:
        raise ValueError(f'Unknown decoder type: {args.decoder_type}')

    return {
        'avg_mse_loss_map': avg_mse_loss_map,
        'avg_mse_loss_mean': avg_mse_loss_mean,
        'avg_ce_loss': avg_ce_loss,
        'correct_rate': correct_rate,
        'pred_log_lhs': pred_log_lhs,
        'associated_log_priors': associated_log_priors,
        'pred_log_posts': np.concatenate(pred_log_posts, axis=0)
    }
