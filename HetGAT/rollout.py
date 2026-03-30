import torch 


class MAHACBuffer:
    """
    pre allocated rollout buffer for MAHAC with per-class trajectories

    all tensors have time as the first dimension: (T, B, ...)
    lstm hidden states are stored per timestep for correct PPO recomputation.
    """

    def __init__(
            self,
            T: int, # rollout len
            B: int, # num envs
            n_scouts: int, 
            n_interc: int, 
            obs_dim_scout: int,
            obs_dim_interc: int,
            state_dim_scout: int,
            state_dim_interc: int,
            action_dim: int, 
            ssn_dim: int, 
            hidden_dim: int,
            device: torch.device,
        ):

        self.T = T
        self.B = B
        self.n_s = n_scouts
        self.n_i = n_interc
        self.device = device
        self.ptr = 0 # next timestep

        # scout tensors
        self.scout_obs = torch.zeros(T, B, n_scouts, obs_dim_scout, device=device)
        self.scout_states = torch.zeros(T, B, n_scouts, state_dim_scout, device=device)
        self.scout_actions = torch.zeros(T, B, n_scouts, action_dim, device=device)
        self.scout_log_probs = torch.zeros(T, B, n_scouts, device=device)
        self.scout_values = torch.zeros(T, B, device=device)

        #interc tensors
        self.interc_obs = torch.zeros(T, B, n_interc, obs_dim_interc, device=device)
        self.interc_states = torch.zeros(T, B, n_interc, state_dim_interc, device=device)
        self.interc_actions = torch.zeros(T, B, n_interc, action_dim, device=device)
        self.interc_log_probs = torch.zeros(T, B, n_interc, device=device)
        self.interc_values = torch.zeros(T, B, device=device)

        # shared tensors
        self.rewards = torch.zeros(T, B, device=device)
        self.dones = torch.zeros(T, B, device=device)
        self.positions = torch.zeros(T, B, n_scouts + n_interc, 2, device=device)
        self.ssn_inputs = torch.zeros(T, B, ssn_dim, device=device)

        # lstm hidden states
        # stored before each forward pass so ppo computation can reinit correctly
        n_hidden = 4
        self.scout_hidden = torch.zeros(T, n_hidden, B*n_scouts, hidden_dim, device=device)
        self.interc_hidden = torch.zeros(T, n_hidden, B*n_interc, hidden_dim, device=device)


    def store(
            self,
            obs_s, state_s, actions_s, log_probs_s, value_s,
            obs_i, state_i, actions_i, log_probs_i, value_i,
            reward, done, positions, ssn_input,
            hidden_s: dict, hidden_i: dict
    ):
        """store one timestep, call w t = self.ptr, then ptr advances"""
    
        t = self.ptr

        self.scout_obs[t] = obs_s
        self.scout_states[t] = state_s
        self.scout_actions[t] = actions_s
        self.scout_log_probs[t] = log_probs_s
        self.scout_values[t] = value_s

        self.interc_obs[t] = obs_i
        self.interc_states[t] = state_i
        self.interc_actions[t] = actions_i
        self.interc_log_probs[t] = log_probs_i
        self.interc_values[t] = value_i

        self.rewards[t] = reward
        self.dones[t] = done
        self.positions[t] = positions
        self.ssn_inputs[t] = ssn_input

        # pack lstm hidden state dict into tensor
        # order: obs_h, obs_c, state_h, state_c
        self.scout_hidden[t,0] = hidden_s['obs_h']
        self.scout_hidden[t,1] = hidden_s['obs_c']
        self.scout_hidden[t,2] = hidden_s['state_h']
        self.scout_hidden[t,3] = hidden_s['state_c']

        self.interc_hidden[t,0] = hidden_i['obs_h']
        self.interc_hidden[t,1] = hidden_i['obs_c']
        self.interc_hidden[t,2] = hidden_i['state_h']
        self.interc_hidden[t,3] = hidden_i['state_c']

        self.ptr += 1

    def get_hidden_scout(self, t: int) -> dict:
        """reconstruct hidden state dict for ts t"""
        return {
            'obs_h': self.scout_hidden[t, 0],
            'obs_c': self.scout_hidden[t, 1],
            'state_h': self.scout_hidden[t, 2],
            'state_c': self.scout_hidden[t, 3]
        }
    
    def get_hidden_interc(self, t: int) -> dict:
        """reconstruct hidden state dict for ts t"""
        return {
            'obs_h': self.interc_hidden[t, 0],
            'obs_c': self.interc_hidden[t, 1],
            'state_h': self.interc_hidden[t, 2],
            'state_c': self.interc_hidden[t, 3]
        } 
    
    def reset(self):
        """call after each ppo update, reset pointer reuses memory"""
        self.ptr = 0



