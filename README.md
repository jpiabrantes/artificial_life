# Environments

There are three environments:
- bacteria colony (not in the paper)
- deadly colony (asexual environment in the paper - I called it deadly here because I had introduced violence)
- sexual colony (sexual env in the paper)

# Algorithms

The E-VDN is the MAEQ algorithm (I think I was calling E-VDN: Multi-Agent Evolutionary-Q at the time).

Run `maeq.py` to train the network.

On the bottom, of maeq.py you will see this:

```python
    from models.base import VDNMixer, VDNMixer_2
    from envs.sexual_colony.sexual_colony import SexualColony
    from envs.sexual_colony.env_config import env_default_config

    config = env_default_config.copy()
    config['greedy_reward'] = True # If you change this to False, it will use the evolutionary reward - which is slower but will converge to the same thing
    env_creator = lambda: SexualColony(config) # replace DeadlyColony here if needed
    env = env_creator()
    q_kwargs = {'conv_sizes': [(32, (3, 3), 1)],
                'fc_sizes': [16],
                'last_fc_sizes': [64, 32],
                'conv_input_shape': env.actor_terrain_obs_shape,
                'fc_input_length': np.prod(env.observation_space.shape) - np.prod(env.actor_terrain_obs_shape),
                'action_space': env.action_space,
                'obs_input_shape': env.observation_space.shape}
    brain_creator = lambda: VDNMixer_2(**q_kwargs)

    ## uncomment this block to get the bigger NN
    # q_kwargs = {'hidden_units': [512, 256, 128], 
    #             'observation_space': env.observation_space,
    #             'action_space': env.action_space}
    # brain_creator = lambda: VDNMixer(**q_kwargs)

    ray.init(local_mode=True)
    trainer = MAEQTrainer(env_creator, brain_creator, population_size=1)
```
    
Change these things to train on the Sexual Colony or Deadly Colony. To use the big NN or the smaller. To use the greedy reward (food colected times the evolutionary reward) or not.

# Replay

With the NN trained to a certain point I would execute a rollout and save the actions taken and etc. I used `interface.py` to play back that rollout and analyse what was going on.
