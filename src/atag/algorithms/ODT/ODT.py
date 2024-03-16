import numpy as np
import torch
import wandb

import argparse
import random
import sys
import os
import pathlib
import json

from .evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from .models.decision_transformer import DecisionTransformer
from .models.mlp_bc import MLPBCModel
from .training.act_trainer import ActTrainer
from .training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class Parameters(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Atag2:
    def __init__(self, env, **parameters):
        self.env = env
        self.variant = Parameters(parameters)


    def experiment(self, exp_prefix='experiment'):
        # device = self.variant.get('device', 'cuda')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_to_wandb = self.variant.get('log_to_wandb', False)
        torch.cuda.empty_cache()
        
        
        env_name, dataset = "browser", "web-app"
        model_type = self.variant['model_type']
        group_name = f'{exp_prefix}-{env_name}-{dataset}'
        exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

        model_dir = './results/models/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)


        max_ep_len = 20
        env_targets = [1600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns

        
        # Override env_targets / set different training target for online decision transformer, following paper
        if self.variant['online_training']:
            env_targets = [1600]  # evaluation conditioning targets
            target_online = 1600

        if model_type == 'bc':
            env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

        # state_dim = self.env.observation_space.shape[0]
        # act_dim = self.env.action_space.shape[0]
        state_dim = self.env.state_dim
        act_dim = self.env.action_dim

        # load dataset
        dataset_path = self.variant['dataset_path']
        trajectories = []
        with open(dataset_path, 'rb') as f:
            trajectories = json.load(f)

        for trajectory in trajectories:
            for key, value in trajectory.items():
                # If the value is a list, convert it to a NumPy array
                trajectory[key] = np.array(value)

        # save all path information into separate lists
        mode = self.variant.get('mode', 'normal')
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print('=' * 50)
        print(f'Starting new experiment: {env_name} {dataset}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)

        K = self.variant['K']
        batch_size = self.variant['batch_size']
        num_eval_episodes = self.variant['num_eval_episodes']
        pct_traj = self.variant.get('pct_traj', 1.)

        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(pct_traj*num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
        
        # Sort trajectories from worst to best and cut to buffer size
        if self.variant['online_training']:
            trajectories = [trajectories[index] for index in sorted_inds]
            trajectories = trajectories[:self.variant['online_buffer_size']]
            num_trajectories = len(trajectories)

        starting_p_sample = p_sample
        def get_batch(batch_size=256, max_len=K):
            # Dynamically recompute p_sample if online training
            if self.variant['online_training']:
                traj_lens = np.array([len(path['observations']) for path in trajectories])
                p_sample = traj_lens / sum(traj_lens)
            else:
                p_sample = starting_p_sample
                
            
            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(batch_size):
                if self.variant['online_training']:
                    traj = trajectories[batch_inds[i]]
                else:
                    traj = trajectories[int(sorted_inds[batch_inds[i]])]
                si = random.randint(0, traj['rewards'].shape[0] - 1)
                # get sequences from dataset
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                a.append(traj['act_probs'][si:si + max_len].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                if 'terminals' in traj:
                    d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                else:
                    d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0., a[-1]], axis=1)
                r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

            s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
            #print('========')
            #print(r)
            #print(rtg)
            return s, a, r, d, rtg, timesteps, mask

        if self.variant['online_training']:
            # If online training, use means during eval, but (not during exploration)
            self.variant['use_action_means'] = True
        
        def eval_episodes(target_rew):
            def fn(model):
                returns, lengths = [], []
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        if model_type == 'dt':
                            ret, length = evaluate_episode_rtg(
                                self.env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                scale=scale,
                                target_return=target_rew/scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                                use_means=self.variant['use_action_means'],
                                eval_context=self.variant['eval_context']
                            )
                        else:
                            ret, length = evaluate_episode(
                                self.env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                target_return=target_rew/scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                    returns.append(ret)
                    lengths.append(length)
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_return_max': np.max(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                }
            return fn


        if model_type == 'dt':
            if self.variant['pretrained_model']:
                model = torch.load(self.variant['pretrained_model'],map_location='cuda:0')
                model.stochastic_tanh = self.variant['stochastic_tanh']
                model.approximate_entropy_samples = self.variant['approximate_entropy_samples']
                model.to(device)

            else:
                model = DecisionTransformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    max_length=K,
                    max_ep_len=max_ep_len*2,
                    hidden_size=self.variant['embed_dim'],
                    n_layer=self.variant['n_layer'],
                    n_head=self.variant['n_head'],
                    n_inner=4*self.variant['embed_dim'],
                    activation_function=self.variant['activation_function'],
                    n_positions=1024,
                    resid_pdrop=self.variant['dropout'],
                    attn_pdrop=self.variant['dropout'],
                    stochastic = self.variant['stochastic'],
                    remove_pos_embs=self.variant['remove_pos_embs'],
                    approximate_entropy_samples = self.variant['approximate_entropy_samples'],
                    stochastic_tanh=self.variant['stochastic_tanh']
                )
        elif model_type == 'bc':
            model = MLPBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                hidden_size=self.variant['embed_dim'],
                n_layer=self.variant['n_layer'],
            )
        else:
            raise NotImplementedError
        
        model = model.to(device=device)
        warmup_steps = self.variant['warmup_steps']
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.variant['learning_rate'],
            weight_decay=self.variant['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
            
        if self.variant['online_training']:
            assert(self.variant['pretrained_model'] is not None), "Must specify pretrained model to perform online finetuning"
            self.variant['use_entropy'] = True
            
        if self.variant['online_training'] and self.variant['target_entropy']:
            # Setup variable and optimizer for (log of) lagrangian multiplier used for entropy constraint
            # We optimize the log of the multiplier b/c lambda >= 0
            log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=device)
            multiplier_optimizer = torch.optim.AdamW(
                [log_entropy_multiplier],
                lr=self.variant['learning_rate'],
                weight_decay=self.variant['weight_decay'],
            )
            # multiplier_optimizer = torch.optim.Adam(
            #     [log_entropy_multiplier],
            #     lr=1e-3
            #     #lr=self.variant['learning_rate'],
            # )
            multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
                multiplier_optimizer,
                lambda steps: min((steps+1)/warmup_steps, 1)
            )
        else:
            log_entropy_multiplier = None
            multiplier_optimizer = None 
            multiplier_scheduler = None

        entropy_loss_fn = None
        """if self.variant['stochastic']:
            if self.variant['use_entropy']:
                if self.variant['target_entropy']:
                    loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.exp(log_entropy_multiplier.detach()) * torch.mean(entropies)
                    target_entropy = -act_dim
                    entropy_loss_fn = lambda entropies: torch.exp(log_entropy_multiplier) * (torch.mean(entropies.detach()) - target_entropy)
                else:
                    loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.mean(entropies)
            else:
                loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg,r, a_log_prob, entropies: -torch.mean(a_log_prob)
        else:
            loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg, r, a_log_prob, entropies: torch.mean((a_hat - a)**2)"""

        loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg, r, a_log_prob, entropies: torch.mean((a_hat - a)**2)

        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=loss_fn,
                log_entropy_multiplier=log_entropy_multiplier,
                entropy_loss_fn=entropy_loss_fn,
                multiplier_optimizer=multiplier_optimizer,
                multiplier_scheduler=multiplier_scheduler,
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        elif model_type == 'bc':
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=loss_fn,
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )

        if log_to_wandb:
            wandb.init(
                entity='rikulehtonen',
                name=exp_prefix,
                group=self.variant['group_name'],
                project='Thesis-results',
                config=self.variant
            )
            # wandb.watch(model)  # wandb has some bug
        if self.variant['eval_only']:
            model.eval()
            eval_fns = [eval_episodes(tar) for tar in env_targets]
            
            for iter_num in range(self.variant['max_iters']):
                logs = {}
                for eval_fn in eval_fns:
                    outputs = eval_fn(model)
                    for k, v in outputs.items():
                        logs[f'evaluation/{k}'] = v

                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
        else:
            if self.variant['online_training']:
                for iter in range(self.variant['max_iters']):
                    # Collect new rollout, using stochastic policy
                    for path_iteration in range(5):
                        ret, length, traj = evaluate_episode_rtg(
                                    self.env,
                                    state_dim,
                                    act_dim,
                                    model,
                                    max_ep_len=max_ep_len,
                                    scale=scale,
                                    target_return=target_online/scale,
                                    mode=mode,
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                                    use_means=False,
                                    return_traj=True
                        )
                        # Remove oldest trajectory, add new trajectory

                        trajectories = trajectories[1:]
                        trajectories.append(traj)

                    # Perform update, eval using deterministic policy 
                    outputs = trainer.train_iteration(num_steps=self.variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                    torch.save(model,os.path.join(model_dir, model_type + '_' + exp_prefix + '.pt'))
                    
                    if log_to_wandb:
                        wandb.log(outputs)
            else:
                for iter in range(self.variant['max_iters']):
                    outputs = trainer.train_iteration(num_steps=self.variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                    if log_to_wandb:
                        wandb.log(outputs)

            torch.save(model,os.path.join(model_dir, model_type + '_' + exp_prefix + '.pt'))

