import os

import pygame
import numpy as np
from PIL import Image
import matplotlib.pylab as plt

pygame.font.init()
TITLE_FONT = pygame.font.SysFont('ubuntu', size=18, bold=True)
FONT = pygame.font.SysFont('ubuntu', size=16)
DPI = 96


class GameRenderer:
    def __init__(self, surface, g_variables):
        self.env = g_variables.env
        self.g_variables = g_variables
        self.surface = surface
        self.width, self.height = self.surface.get_size()

    def render(self, iter_):
        dict_ = self.g_variables.dicts[iter_]
        state = dict_['state']
        agent_dict = dict_['agents'][self.g_variables.following_agent_id]
        agent_row, agent_col = agent_dict['row'], agent_dict['col']
        img = self.env.render(state=state)
        if self.g_variables.vision_mask:
            vision_grid = np.arange(1 + 2 * self.env.vision) - self.env.vision
            mask = np.ones((img.shape[0], img.shape[1]), np.bool)
            rows = np.mod(agent_row + vision_grid, self.env.n_rows)
            cols = np.mod(agent_col + vision_grid, self.env.n_cols)
            mask[np.ix_(rows, cols)] = False
            img[mask, :] = 0.5*img[mask, :]+0.5*np.array((0, 0, 0), np.float)
        if self.g_variables.tracking:
            zoom = self.g_variables.zoom
            grid = np.arange(1 + 2 * zoom) - zoom
            rows = np.mod(agent_row + grid, self.env.n_rows)
            cols = np.mod(agent_col + grid, self.env.n_cols)
            img = img[np.ix_(rows, cols)]
        img = (img*255).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize((self.width, self.height), Image.NEAREST))
        pygame.surfarray.blit_array(self.surface, img.transpose(1, 0, 2))


class PolicyRenderer:
    def __init__(self, surface, g_variables):
        self.g_variables = g_variables
        self.margin = 10
        width, height = surface.get_size()
        self.surface = surface.subsurface(self.margin, self.margin, width-2*self.margin, height-2*self.margin)
        self.width, self.height = self.surface.get_size()
        self.down_arrow = pygame.image.load(os.path.join('assets', 'down_arrow.png'))
        self.still = pygame.image.load(os.path.join('assets', 'still.png'))
        self.background = self._build_background()

    def _build_background(self):
        title_surf = TITLE_FONT.render('Policy:', True, (0, 0, 0))
        self.surface.blit(title_surf, (0, 0))

        # Icons
        y = title_surf.get_height() + self.margin
        icon_width = self.down_arrow.get_width()
        gap_size = (self.width-icon_width*5)/6
        assert gap_size > 0, 'there is not enough width for the Policy Renderer'
        self.x_centers = [gap_size*(i+1)+icon_width*(i+0.5) for i in range(5)]
        [self.surface.blit(pygame.transform.rotate(self.down_arrow, 90*i), (x-icon_width/2, y))
         for i, x in enumerate(self.x_centers[:-1])]
        self.surface.blit(self.still, (self.x_centers[-1]-icon_width/2, y))
        self.y = y + self.down_arrow.get_height() + self.margin
        return self.surface.copy()

    def render(self, iter_):
        dict_ = self.g_variables.dicts[iter_]
        self.surface.blit(self.background, (0, 0))
        action_probs = dict_['agents'][self.g_variables.following_agent_id]['action_probs']
        surfaces = [FONT.render('%.1f%%' % (prob*100), True, (0, 0, 0)) for prob in action_probs]
        widths = [surf.get_width() for surf in surfaces]
        [self.surface.blit(surf, (x-width/2, self.y)) for surf, width, x in zip(surfaces, widths, self.x_centers)]


class BacteriaAttributeRenderer:
    def __init__(self, surface, g_variables):
        self.g_variables = g_variables
        self.margin = 10
        width, height = surface.get_size()
        self.surface = surface.subsurface(self.margin, self.margin, width-2*self.margin, height-2*self.margin)
        self.width, self.height = self.surface.get_size()
        self.background = surface.copy()

    def render(self, iter_):
        env = self.g_variables.env
        dict_ = self.g_variables.dicts[iter_]
        self.surface.blit(self.background, (0, 0))

        # title
        followed_surf = TITLE_FONT.render('Following %s' % self.g_variables.following_agent_id, True, (0, 0, 0))
        self.surface.blit(followed_surf, (0, 0))
        population_surf = TITLE_FONT.render('Total population: %d' % len(dict_['agents']), True, (0, 0, 0))
        self.surface.blit(population_surf, (self.width-population_surf.get_width(), 0))

        # Attributes
        state = dict_['state']
        agent_dict = dict_['agents'][self.g_variables.following_agent_id]

        agent_dict['family_size'] = np.sum(state[:, :, env.State.DNA] == agent_dict['dna'])
        agent_dict['total_population'] = np.sum(state[:, :, env.State.AGENTS])
        agent_dict['iter'] = iter_

        labels = ['Iteration', 'Age', 'Sugar', 'Family size', 'Total population']
        keys = ['iter', 'age', 'sugar', 'family_size', 'total_population']

        # render labels
        label_surfaces = [FONT.render(label, True, (0, 0, 0)) for label in labels]
        label_widths = [surf.get_width() for surf in label_surfaces]
        gap_size = (self.width-sum(label_widths))/(len(label_widths)+1)
        assert gap_size > 0, 'labels dont fit on the available surface'

        key_surfaces = [FONT.render('%d' % agent_dict[key] if type(agent_dict[key]) is int
                                     else '%.1f' % agent_dict[key], True, (0, 0, 0)) for key in keys]
        key_widths = [surf.get_width() for surf in key_surfaces]

        # blit
        y = population_surf.get_height()+self.margin
        x = gap_size
        x_centers = []
        for surf, width in zip(label_surfaces, label_widths):
            self.surface.blit(surf, (x, y))
            x_centers.append(x+width//2)
            x += width + gap_size

        y += label_surfaces[0].get_height()+self.margin
        x = 0
        for surf, width, x_center in zip(key_surfaces, key_widths, x_centers):
            self.surface.blit(surf, (x_center - width//2, y))
            x += width


class FamilyRenderer:
    def __init__(self, surface, g_struct):
        self.g_struct = g_struct
        self.margin = 10
        width, height = surface.get_size()
        self.surface = surface.subsurface(self.margin, self.margin, width-2*self.margin, height-2*self.margin)
        self.width, self.height = self.surface.get_size()
        self.background = self.surface.copy()
        self.family_sizes = self._compute_family_sizes()
        self.fig, self.axs, self.line = None, None, None
        self.colors = [(200, 0, 0), (0, 0, 200), (0, 100, 100), (0, 0, 0), (200, 200, 200)]
        self.colors = [np.array(c, np.float32)/255 for c in self.colors]

    def _compute_family_sizes(self):
        env = self.g_struct.env
        n_iters = len(self.g_struct.dicts)
        n_families = len(np.unique(self.g_struct.dicts[0]['state'][..., env.State.DNA]))-1
        family_sizes = np.empty((n_families, n_iters), np.int)
        for i in range(n_iters):
            state = self.g_struct.dicts[i]['state']
            for dna in range(1, n_families+1):
                family_sizes[dna-1, i] = np.sum(state[..., env.State.DNA] == dna)
        return family_sizes

    def render(self, iter_):
        self.surface.blit(self.background, (0, 0))
        if self.fig is None:
            self.fig, self.axs = plt.subplots(figsize=(self.width // DPI, self.height // DPI))
            self.axs.stackplot(np.arange(self.family_sizes.shape[1]), self.family_sizes, colors=self.colors)
            self.line = self.axs.axvline(x=iter_)
            self.axs.set_xlim([0, self.family_sizes.shape[1]])
            self.axs.set_ylabel('Family sizes')
        else:
            self.line.set_xdata(iter_)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        pygame.surfarray.blit_array(self.surface, fig2rgb(self.fig, self.width, self.height))


def fig2rgb(fig, width, height):
    img1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = Image.frombuffer('RGB', fig.canvas.get_width_height(), img1).resize((width, height), Image.BILINEAR)
    img = np.array(img).transpose(1, 0, 2)
    img = img[:, ::-1, :]
    return img

