import numpy as np


def get_states_actions_for_locs_and_dna(state_action, locs, dnas, n_rows, n_cols, dna_index):
    states_actions = np.empty((len(locs), ) + state_action.shape)
    # center state and action maps on agent
    for i, ((row, col), dna) in enumerate(zip(locs, dnas)):
        states_actions[i] = _center_state_action_on_loc(state_action, row, col, n_rows, n_cols)
        states_actions[i][..., dna_index] = states_actions[i][..., dna_index] == dna
    # Erase the action of the centered agent
    states_actions[:, n_rows // 2, n_cols // 2, -1] = -1
    return states_actions


def _center_state_action_on_loc(state_action, row, col, n_rows, n_cols):
    rows = np.mod(row+(np.arange(n_rows) - n_rows//2), n_rows)
    cols = np.mod(col+(np.arange(n_cols) - n_cols//2), n_cols)
    indices = np.ix_(rows, cols)
    return state_action[indices]
