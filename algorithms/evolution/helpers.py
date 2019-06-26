import os
import pickle


def load_variables(env, generation=None):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(file_dir, 'checkpoints', env.name)
    if generation is None:
        with open(os.path.join(checkpoint_path, 'last_checkpoint.txt')) as f:
            last_generation = int(f.readline())
        print('Loaded data from generation {}'.format(last_generation))
    else:
        last_generation = generation
    with open(os.path.join(checkpoint_path, str(last_generation) + '_variables.pkl'), 'rb') as f:
        mu0, stds, horizons_list, returns_list, filters = pickle.load(f)
    return last_generation, mu0, stds, filters
