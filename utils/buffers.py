from collections import defaultdict

import numpy as np

import utils.misc as misc


class EpStats:
    def __init__(self):
        self._stats = defaultdict(list)

    def add(self, stats):
        for k, v in stats.items():
            self._stats[k].append(v)

    def get(self):
        result = self._stats
        self._stats = defaultdict(list)
        return result


class COMABuffer:
    def __init__(self, size, observation_space, action_space, gamma, lamb):
        self.obs_buf = np.empty((size,) + observation_space.shape, dtype=np.float32)
        self.act_buf = np.empty((size,) + action_space.shape, dtype=np.int32)
        self.adv_buf = np.empty(size, dtype=np.float32)
        self.rew_buf = np.empty(size, dtype=np.float32)
        self.q_buf = np.empty(size, dtype=np.float32)
        self.log_probs_buf = np.empty(size, dtype=np.float32)
        self.pi_buf = np.empty((size, action_space.n), dtype=np.float32)
        self.loc_buf = np.empty((size, 2), dtype=np.int32)
        self.dna_buf = np.empty(size, dtype=np.int32)
        self.ret_buf = np.empty(size, dtype=np.float32)
        self.ind_buf = np.empty(size, dtype=np.int32)
        self.path_start_idx, self.ptr, self.max_size = 0, 0, size
        self.gamma, self.lamb = gamma, lamb

    def store(self, obs, act, rew, q, adv, log_p, pi, loc, dna, ind):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.q_buf[self.ptr] = q
        self.adv_buf[self.ptr] = adv
        self.log_probs_buf[self.ptr] = log_p
        self.pi_buf[self.ptr] = pi
        self.loc_buf[self.ptr] = loc
        self.dna_buf[self.ptr] = dna
        self.ind_buf[self.ptr] = ind
        self.ptr += 1

    def finnish_path(self, last_value):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        #vals = np.append(self.val_buf[path_slice], last_value)

        # the next two lines implement TTD(Lambda) calculation. 12.3 Intro to RL
        #delta = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        #self.td_buf[path_slice] = vals[:-1] + misc.discount_cumsum(delta, self.gamma * self.lamb)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = misc.discount_cumsum(rews[:-1], self.gamma)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        to_idx = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        return (self.obs_buf[:to_idx], self.act_buf[:to_idx], self.adv_buf[:to_idx], self.ret_buf[:to_idx],
                self.log_probs_buf[:to_idx], self.pi_buf[:to_idx], self.q_buf[:to_idx], self.loc_buf[:to_idx],
                self.dna_buf[:to_idx], self.ind_buf[:to_idx])


class CentralPPOBuffer:
    def __init__(self, size, observation_space, action_space, states_actions_shape, gamma, lamb):
        self.obs_buf = np.empty((size,) + observation_space.shape, dtype=np.float32)
        self.c_obs_buf = np.empty((size,) + states_actions_shape, dtype=np.float32)
        self.act_buf = np.empty((size,) + action_space.shape, dtype=np.int32)
        self.adv_buf = np.empty(size, dtype=np.float32)
        self.rew_buf = np.empty(size, dtype=np.float32)
        self.ret_buf = np.empty(size, dtype=np.float32)
        self.val_buf = np.empty(size, dtype=np.float32)
        self.log_probs_buf = np.empty(size, dtype=np.float32)
        self.path_start_idx, self.ptr, self.max_size = 0, 0, size
        self.gamma, self.lamb = gamma, lamb

    def store(self, obs, act, rew, val, logp, state_action):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.log_probs_buf[self.ptr] = logp
        self.c_obs_buf[self.ptr] = state_action
        self.ptr += 1

    def finnish_path(self, last_value):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = misc.discount_cumsum(deltas, self.gamma * self.lamb)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = misc.discount_cumsum(rews[:-1], self.gamma)
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        to_idx = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        return (self.obs_buf[:to_idx], self.act_buf[:to_idx], self.adv_buf[:to_idx], self.ret_buf[:to_idx],
                self.log_probs_buf[:to_idx], self.c_obs_buf[:to_idx])



class PPOBuffer:
    def __init__(self, size, observation_space, action_space, gamma, lamb):
        self.obs_buf = np.empty((size,) + observation_space.shape, dtype=np.float32)
        self.act_buf = np.empty((size,) + action_space.shape, dtype=np.int32)
        self.adv_buf = np.empty(size, dtype=np.float32)
        self.rew_buf = np.empty(size, dtype=np.float32)
        self.ret_buf = np.empty(size, dtype=np.float32)
        self.val_buf = np.empty(size, dtype=np.float32)
        self.log_probs_buf = np.empty(size, dtype=np.float32)
        self.path_start_idx, self.ptr, self.max_size = 0, 0, size
        self.gamma, self.lamb = gamma, lamb

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.log_probs_buf[self.ptr] = logp
        self.ptr += 1

    def finnish_path(self, last_value):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = misc.discount_cumsum(deltas, self.gamma * self.lamb)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = misc.discount_cumsum(rews[:-1], self.gamma)
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        to_idx = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        return (self.obs_buf[:to_idx], self.act_buf[:to_idx], self.adv_buf[:to_idx], self.ret_buf[:to_idx],
                self.log_probs_buf[:to_idx])
