import os
from collections import defaultdict

import pandas as pd

features = ['species_index', 'dna', 'health', 'sugar', 'age', 'family_size']
columns = ['%s_%d' % (key, i) for i in range(1, 3) for key in features] + ['compete_1', 'compete_2', 'total_population']


class CompetitiveScenarios:
    def __init__(self):
        self.data = []
        self.best_tiles = defaultdict(list)

    def add_best_tile(self, agent_dict, location):
        self.best_tiles[location].append(agent_dict)

    def add_rows(self, actions_dict):
        for location, agents_list in self.best_tiles.items():
            if len(agents_list) == 1:
                continue
            competitions = []
            for i in range(len(agents_list)-1):
                agent_1 = agents_list[i]
                row1, col1 = agent_1['row'], agent_1['col']
                for j in range(i+1, len(agents_list)):
                    agent_2 = agents_list[j]
                    assert agent_1['id'] != agent_2['id']
                    row2, col2 = agent_2['row'], agent_2['col']
                    if (row1-row2, col1-col2) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        competitions.append((agent_1, agent_2))
            for agent_1, agent_2 in competitions:
                # if there is only a competing agent, it will always be the second agent
                # if self._is_competing(location, (agent_1['row'], agent_1['col']), actions_dict[agent_1['id']]):
                #     tmp = agent_1
                #     agent_1 = agent_2
                #     agent_2 = tmp
                row = [agent_1[key] for key in features]
                row += [agent_2[key] for key in features]
                row += [self._is_competing(location, (agent_1['row'], agent_1['col']), actions_dict[agent_1['id']])]
                row += [self._is_competing(location, (agent_2['row'], agent_2['col']), actions_dict[agent_2['id']])]
                row += [len(actions_dict)]
                self.data.append(row)
        self.best_tiles = defaultdict(list)

    def save(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        results = pd.DataFrame(self.data, columns=columns)
        path = os.path.join(cwd, 'data', 'vdn_competitive.csv')
        if os.path.isfile(path):
            results.to_csv(path, mode='a', header=False)
        else:
            results.to_csv(path)

    def _is_competing(self, best_tile_loc, agent_loc, agent_action):
        row, col = agent_loc
        movement = agent_action % 5
        if movement != 4:  # if agent moving
            if movement == 0:
                row += 1
            elif movement == 1:
                col += 1
            elif movement == 2:
                row -= 1
            elif movement == 3:
                col -= 1
            row, col = row % 50, col % 50
        return best_tile_loc == (row, col)
