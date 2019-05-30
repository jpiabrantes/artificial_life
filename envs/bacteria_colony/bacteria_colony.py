from copy import deepcopy
from time import time

import numpy as np
# from cv2 import cvtColor, COLOR_HSV2RGB
from gym.spaces import Box, Discrete

from utils.map import create_tiles_1
from utils.misc import Enum, MeanTracker, Timer

State = Enum(('SUGAR', 'AGENTS', 'AGE', 'AGENT_SUGAR', 'DNA'))
Terrain = Enum(('SUGAR', 'AGENTS', 'AGE', 'AGENT_SUGAR', 'KINSHIP'))
DNA_COLORS = {1: (200, 0, 0), 2: (0, 0, 200), 3: (0, 100, 100), 4: (0, 0, 0), 5: (255, 255, 255)}


class BacteriaColony:
    name = 'BacteriaColony-v0'

    metadata = {'render.modes': ['rgb_array']}
    reward_range = (0.0, float('inf'))

    def __init__(self, config):
        # env parameters
        self.n_rows = config['n_rows']
        self.n_cols = config['n_cols']
        self.sugar_growth_rate = config['sugar_growth_rate']

        self.greedy_reward = config['greedy_reward']
        self.n_agents = config['n_agents']
        self.birth_endowment = config['birth_endowment']
        self.metabolism = config['metabolism']
        self.infertility_age = config['infertility_age']
        self.fertility_age = config['fertility_age']
        self.longevity = config['longevity']
        self.max_iters = config['max_iters']
        self.max_capacity = config['max_capacity']

        vision = config['vision']
        assert type(vision) is int, 'Vision needs to be an integer'
        assert vision < min(self.n_rows//2, self.n_cols//2), "Vision cant be larger than half the grid world"
        self.vision = vision

        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=1, shape=((vision * 2 + 1)**2 * len(Terrain) + 4,),
                                     dtype=np.float32)
        self.actor_terrain_obs_shape = (vision * 2 + 1, vision * 2 + 1, len(Terrain))
        self.critic_observation_shape = (self.n_rows, self.n_cols, len(State))

        # episode parameters
        self._state = None
        self._capacity = create_tiles_1(self.n_rows, self.n_cols, self.max_capacity)
        self.agents = None
        self.tiles = None
        self.iter = None

        # Tracking metrics
        self.timers = {'move': MeanTracker(), 'mate_collect_eat': MeanTracker(), 'observe': MeanTracker()}
        self.babies_born = None
        self.surplus = None
        self.life_expectancy = None
        self.dna_total_score = None
        self.average_population = None

    @staticmethod
    def seed(seed=None):
        if seed is None:
            seed = int(time())
        np.random.seed(seed)
        return seed

    def reset(self, species_indices=None):
        """
        Resets the env and returns observations from ready agents.

        :return: obs (dict): New observations for each ready agent.
        """
        if species_indices is None:
            species_indices = list(range(self.n_agents))
        Agent.id = 1
        self.babies_born = 0
        self.average_population = MeanTracker()
        self.dna_total_score = [0]*self.n_agents
        self.surplus = MeanTracker()
        self.life_expectancy = MeanTracker()
        # this will be filled by the agents (so that later newborns can add themselves to the world)
        self.agents = {}  # {agent.id: agent}

        self._state = np.zeros((self.n_rows, self.n_cols, len(State)), np.float32)
        self._state[:, :, State.SUGAR] = self._capacity.copy()

        self.tiles = self._create_tiles()
        self._create_agents(species_indices)
        self.iter = 0
        return self._generate_observations()

    def step(self, action_dict):
        """
        Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        :param action_dict: Action for each ready agent.
        :return:
        obs_dict (dict): New observations for each ready agent.
        reward_dict (dict): Reward values for each ready agent. If the
            episode is just started, the value will be None.
        done_dict (dict): Done values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
        info_dict (dict): Optional info values for each agent id.

        Timeline:
        - Agents act
        - Mate, collect and eat
        - Agents (including children) observe
        """
        done_dict, reward_dict, info_dict = {}, {}, {}

        # randomise the order in which actions are taken
        action_items = list(action_dict.items())
        np.random.shuffle(action_items)

        # Agents act: move, harvest, eat, reproduce and age
        dna_map = self._state[:, :, State.DNA].copy()
        self.agent_dna = {}  # Keep a record of an agent dna (important if it dies)
        with Timer() as move_timer:
            for agent_name, action in action_items:
                agent = self.agents[agent_name]
                self.agent_dna[agent_name] = agent.dna
                newborn, harvest, died = agent.step(action, self.n_rows, self.n_cols, self.tiles)
                if self.greedy_reward:
                    reward_dict[agent_name] = harvest
                if newborn:
                    self.babies_born += 1
                if died:
                    self.life_expectancy.add_value(agent.age)
                    agent.die()
                    del self.agents[agent_name]
                    done_dict[agent_name] = True
                else:
                    done_dict[agent_name] = False
        self.timers['move'].add_value(move_timer.interval)

        # Compute surplus stat
        self.average_population.add_value(len(self.agents))
        # Which tiles would feed another agent
        mask = self._state[:, :, Terrain.SUGAR] > self.metabolism
        # For how many iters would they feed the current population?
        if len(self.agents):
            self.surplus.add_value(np.sum(self._state[mask, Terrain.SUGAR])/len(self.agents))

        self._grow_sugar()
        with Timer() as observe_timer:
            obs_dict = self._generate_observations()
        self.timers['observe'].add_value(observe_timer.interval)

        if not self.greedy_reward:
            # compute a reward for every agent that sent an action
            dna_results = np.zeros((self.n_agents,), np.int)
            dnas, counts = np.unique(dna_map[dna_map != 0], return_counts=True)
            for dna, count in zip(dnas, counts):
                dna_results[int(dna) - 1] = count
                self.dna_total_score[int(dna) - 1] += count
            for agent_name, dna in self.agent_dna.items():
                reward_dict[agent_name] = dna_results[agent.dna - 1]

        # Check termination
        self.iter += 1
        if (len(self.agents) == 0) or (self.iter == self.max_iters):
            done_dict['__all__'] = True
            info_dict['__all__'] = {'surplus': self.surplus.mean, 'babies_born': self.babies_born,
                                    'survivors': len(self.agents), 'life_expectancy': self.life_expectancy.mean,
                                    'average_population': self.average_population.mean}
            if not self.greedy_reward:
                info_dict['founders_results'] = dna_results
                info_dict['founders_total_results'] = self.dna_total_score
        else:
            done_dict['__all__'] = False

        return obs_dict, reward_dict, done_dict, info_dict

    def render(self, state=None, mode='rgb_array'):
        if state is None:
            state = self._state
        if mode == 'rgb_array':
            # agents
            a_rgb = np.zeros((self.n_rows, self.n_cols, 3), np.float32)
            for dna in range(1, 6):
                mask = state[:, :, State.DNA] == dna
                a_rgb[mask, :] = np.array(DNA_COLORS[dna], dtype=np.float32)/255.

            # sugar
            dirt = np.array((120., 72, 0)) / 255
            grass = np.array((85., 168, 74)) / 255
            norm_sugar = (state[:, :, State.SUGAR]/self.max_capacity)[..., None]
            s_rgb = norm_sugar*grass+(1-norm_sugar)*dirt
            mask = state[..., State.AGENTS] > 0
            s_rgb[mask] = a_rgb[mask]
            return s_rgb
        else:
            super(BacteriaColony, self).render(mode)

    def get_kinship_map(self, agent_name):
        """
        Returns a kinship 2D map in respect to the requested agent.
        Note: This will be called by the optimisation algorithm.

        :param agent_name: string
        :return: array (n_rows, n_cols)
        """
        dna = self.agent_dna[agent_name]
        return self._state[:, :, State.DNA] == dna

    def to_dict(self):
        agents_dict = {}
        for agent in self.agents.values():
            agents_dict[agent.id] = agent.to_dict()
        result = {'agents': agents_dict,
                  'state': self._state.copy(),
                  'iter': self.iter,
                  }
        return result

    def add_agent_callback(self, agent):
        self.agents[agent.id] = agent

    def _generate_observations(self):
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.tiles[row, col].update_state()
        obs_dict = {'state': self._state.copy()}
        for key, agent in self.agents.items():
            grid_size = 1 + 2 * self.vision
            vision = np.arange(grid_size) - self.vision
            rows = np.mod(agent.row + vision, self.n_rows)
            cols = np.mod(agent.col + vision, self.n_cols)
            obs = self._state[np.ix_(rows, cols)].copy()  # TODO: test the copy
            kinship_grid = self._state[:, :, State.DNA] == agent.dna
            obs[:, :, Terrain.KINSHIP] = kinship_grid[np.ix_(rows, cols)]

            family_size = kinship_grid.sum()
            n_entities = len(self.agents)
            fc_obs = np.array((family_size, agent.row, agent.col, n_entities))
            obs_dict[key] = np.hstack((obs.ravel(), fc_obs))
        return obs_dict

    def _create_agents(self, species_indices):
        locations = set()
        for i in range(self.n_agents):
            while True:
                row = np.random.randint(-10, 11)
                col = self.n_cols//2 + int(np.round(np.sqrt(100-row**2)))*np.random.choice((-1, 1))
                row += self.n_rows//2
                if (row, col) not in locations:
                    locations.add((row, col))
                    break
            tile = self.tiles[row, col]
            Agent(row, col, self.birth_endowment, self.metabolism, self.fertility_age, self.infertility_age,
                  self.longevity, tile, self.add_agent_callback, species_indices[i])

    def _grow_sugar(self):
        self._state[:, :, Terrain.SUGAR] = np.minimum(self._state[:, :, Terrain.SUGAR]
                                                      + self.sugar_growth_rate, self._capacity)

    def _create_tiles(self):
        def create_tile(row, col):
            return Tile(row, col, self._state[row, col, :])
        tiles = dict()
        tiles[0, 0] = create_tile(0, 0)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                tile = tiles[r, c]
                for row, col in [(r, c-1), (r-1, c), (r, c+1), (r+1, c)]:
                    row = row % self.n_rows
                    col = col % self.n_cols
                    if (row, col) in tiles:
                        neighbour = tiles[row, col]
                    else:
                        neighbour = create_tile(row, col)
                        tiles[row, col] = neighbour
                    tile.neighbours.append(neighbour)
        return tiles


class Tile:
    """
    Each tile is responsible to keep track of its state. When doing so it will keep the global world state and farms
    dict up to date.

    *Note:* It would make sense for each Tile to also be responsible to grow its own sugar but currently it seems easier
    to do this in the world since we can vectorise the growth of sugar in all tiles.
    """
    def __init__(self, row, col, state_at_tile):
        self.agent = None
        self.row = row
        self.col = col
        self.neighbours = []
        self._state_at_tile = state_at_tile

    def harvest(self, agent):
        """
        returns how much sugar an agent can harvest on this tile.

        *Note* The harvested sugar will be removed in the world and it will grow also in the world.
        :param agent:
        :return: sugar_to_harvest
        """
        assert agent == self.agent, "Agent is trying to harvest from the wrong tile"
        harvest = self._state_at_tile[Terrain.SUGAR]
        self._state_at_tile[Terrain.SUGAR] = 0
        return harvest

    def update_state(self):
        if self.agent:
            self._state_at_tile[State.AGENT_SUGAR] = self.agent.sugar
            self._state_at_tile[State.AGE] = self.agent.age

    def find_available_tile(self):
        for tile in self.neighbours:
            if tile.agent is None:
                return tile
        return None

    def add_agent(self, agent):
        if self.agent:  # tile already occupied
            assert agent != self.agent, 'agent trying to enter same tile twice'
            return False
        self.agent = agent
        self._state_at_tile[Terrain.AGENTS] += 1
        self._state_at_tile[State.DNA] = self.agent.dna
        return True

    def remove_agent(self, agent):
        assert agent == self.agent, 'Agent is trying to be removed from the wrong tile'
        self.agent = None
        self._state_at_tile[State.DNA] = 0
        self._state_at_tile[State.AGENTS] -= 1


class Agent:
    id = 1

    def __init__(self, row, col, endowment, metabolism, fertility_age, infertility_age, longevity, tile,
                 bring_agent_to_world_fn, species_index, dna=None):
        self.dna = Agent.id if dna is None else dna
        self.species_index = 0 if species_index is None else species_index
        self.id = '{}_{:d}'.format(species_index, Agent.id)
        Agent.id += 1
        self.row = row
        self.col = col
        self.endowment = endowment
        self.sugar = endowment
        self.metabolism = metabolism
        self.tile = tile
        self.bring_agent_to_world_fn = bring_agent_to_world_fn
        self.longevity = longevity
        self.fertility_age = fertility_age
        self.infertility_age = infertility_age

        self.age = 0
        bring_agent_to_world_fn(self)
        tile.add_agent(self)

    def to_dict(self):
        to_delete = {'bring_agent_to_world_fn', 'tile'}
        to_store = set(self.__dict__.keys()).difference(to_delete)
        return {k: deepcopy(self.__dict__[k]) for k in to_store}

    def _reproduce(self):
        if self.is_fertile:
            tile = self.tile.find_available_tile()
            if tile:
                newborn = Agent(tile.row, tile.col, self.endowment, self.metabolism, self.fertility_age,
                                self.infertility_age, self.longevity, tile, self.bring_agent_to_world_fn,
                                self.species_index, self.dna)
                return newborn
        return None

    @property
    def is_fertile(self):
        return (self.age >= self.fertility_age) and (self.age < self.infertility_age) and \
               (self.sugar > self.endowment*2)

    def die(self):
        self.tile.remove_agent(self)

    def step(self, action, n_rows, n_cols, tiles):
        assert action in range(5), "Action needs to be an int between 0 and 4"
        if action != 4:  # if agent moving
            new_row, new_col = self.row, self.col
            if action == 0:
                new_row += 1
            elif action == 1:
                new_col += 1
            elif action == 2:
                new_row -= 1
            elif action == 3:
                new_col -= 1
            new_row = new_row % n_rows
            new_col = new_col % n_cols
            new_tile = tiles[new_row, new_col]
            if new_tile.add_agent(self):
                self.tile.remove_agent(self)
                self.tile = new_tile
                self.row, self.col = new_row, new_col

        harvest = self.tile.harvest(self)
        self.sugar += harvest - self.metabolism
        newborn = self._reproduce()
        if newborn:
            self.sugar -= self.endowment
        self.age += 1
        died = (self.sugar < 0) or (self.age == self.longevity)
        return newborn, harvest, died
