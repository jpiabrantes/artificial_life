import os

import pygame
import numpy as np
from PIL import Image
import matplotlib.pylab as plt

pygame.font.init()
TITLE_FONT = pygame.font.SysFont('ubuntu', size=18, bold=True)
FONT = pygame.font.SysFont('ubuntu', size=16)
DPI = 96
plt.style.use('bmh')


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
        track_location = (agent_dict['row'], agent_dict['col'])

        tracking_dict = {'vision_mask': self.g_variables.vision_mask, 'tracking': self.g_variables.tracking,
                         'track_location': track_location, 'zoom': self.g_variables.zoom}

        img = self.env.render(state=state, tracking_dict=tracking_dict)
        img = np.array(Image.fromarray(img).resize((self.width, self.height), Image.NEAREST))
        pygame.surfarray.blit_array(self.surface, img.transpose(1, 0, 2))
        #pygame.image.save(self.surface, 'images/%d.png' % iter_)


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
        gap_size = (self.width-icon_width*7)/8
        assert gap_size > 0, 'there is not enough width for the Policy Renderer'
        self.x_centers = [gap_size*(i+1)+icon_width*(i+0.5) for i in range(7)]
        [self.surface.blit(pygame.transform.rotate(self.down_arrow, 90*i), (x-icon_width/2, y))
         for i, x in enumerate(self.x_centers[:4])]
        [self.surface.blit(self.still, (x-icon_width/2, y)) for x in self.x_centers[4:]]
        self.y = y + self.down_arrow.get_height() + self.margin
        return self.surface.copy()

    def render(self, iter_):
        dict_ = self.g_variables.dicts[iter_]
        self.surface.blit(self.background, (0, 0))
        action_probs = dict_['agents'][self.g_variables.following_agent_id]['action_probs']
        action_probs = action_probs.reshape(2, 5)
        action_probs = action_probs.sum(axis=0).tolist()+[action_probs[0, :].sum()] + [action_probs[1, :].sum()]

        surfaces = [FONT.render('%.1f%%' % (prob*100), True, (0, 0, 0)) for prob in action_probs]
        widths = [surf.get_width() for surf in surfaces]
        [self.surface.blit(surf, (x-width/2, self.y)) for surf, width, x in zip(surfaces, widths, self.x_centers)]


class BacteriaAttributeRenderer:
    def __init__(self, surface, g_variables):
        self.g_variables = g_variables
        self.margin = 10
        self.y_margin = 20
        width, height = surface.get_size()
        self.surface = surface.subsurface(self.margin, self.margin, width-2*self.margin, height-2*self.margin)
        self.width, self.height = self.surface.get_size()
        self.background = surface.copy()

    def render(self, iter_):
        env = self.g_variables.env
        dict_ = self.g_variables.dicts[iter_]
        self.surface.blit(self.background, (0, 0))

        # title
        species_idx, agent_idx = self.g_variables.following_agent_id.split('_')
        followed_surf = TITLE_FONT.render('Following agent: %s from species: %s' % (agent_idx, species_idx),
                                          True, (0, 0, 0))
        self.surface.blit(followed_surf, (0, 0))
        population_surf = TITLE_FONT.render('Total population: %d' % len(dict_['agents']), True, (0, 0, 0))
        self.surface.blit(population_surf, (self.width-population_surf.get_width(), 0))

        # Attributes
        state = dict_['state']
        agent_dict = dict_['agents'][self.g_variables.following_agent_id]

        agent_dict['family_size'] = np.sum(state[:, :, env.State.DNA] == agent_dict['dna'])
        agent_dict['iter'] = iter_

        labels = ['Iteration', 'Age', 'Food stored', 'Health', 'Family size']
        keys = ['iter', 'age', 'sugar', 'health', 'family_size']

        # render labels
        label_surfaces = [FONT.render(label, True, (0, 0, 0)) for label in labels]
        label_widths = [surf.get_width() for surf in label_surfaces]
        gap_size = (self.width-sum(label_widths))/(len(label_widths)+1)
        assert gap_size > 0, 'labels dont fit on the available surface'

        key_surfaces = [FONT.render('%d' % agent_dict[key] if type(agent_dict[key]) is int
                                     else '%.1f' % agent_dict[key], True, (0, 0, 0)) for key in keys]
        key_widths = [surf.get_width() for surf in key_surfaces]

        # blit
        y = population_surf.get_height()+self.y_margin
        x = gap_size
        x_centers = []
        for surf, width in zip(label_surfaces, label_widths):
            self.surface.blit(surf, (x, y))
            x_centers.append(x+width//2)
            x += width + gap_size

        y += label_surfaces[0].get_height()+self.y_margin
        x = 0
        for surf, width, x_center in zip(key_surfaces, key_widths, x_centers):
            self.surface.blit(surf, (x_center - width//2, y))
            x += width


class FamilyRenderer:
    def __init__(self, surface, g_struct, rotate=False):
        self.rotate = rotate
        self.g_struct = g_struct
        self.margin = 10
        width, height = surface.get_size()
        self.surface = surface.subsurface(self.margin, self.margin, width-2*self.margin, height-2*self.margin)
        self.width, self.height = self.surface.get_size()
        self.background = self.surface.copy()
        self.family_sizes = self._compute_family_sizes()
        self.fig, self.axs, self.line = None, None, None
        yellow, cyan, mangenta, black, white
        self.colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 0), (200, 200, 200)]
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
            if self.rotate:
                self.fig, self.axs = plt.subplots(figsize=(self.height // DPI, self.width // DPI))
            else:
                self.fig, self.axs = plt.subplots(figsize=(self.width // DPI, self.height // DPI))

            self.axs.stackplot(np.arange(self.family_sizes.shape[1]), self.family_sizes, colors=self.colors)
            self.line = self.axs.axvline(x=iter_)
            self.axs.set_xlim([0, self.family_sizes.shape[1]])
            self.axs.set_ylabel('Family size', fontsize=18)
            self.axs.set_xlabel('Iteration', fontsize=18, rotation=180*self.rotate)
            if self.rotate:
                plt.xticks(rotation=90)
                plt.yticks(rotation=90)
        else:
            self.line.set_xdata(iter_)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        img = fig2rgb(self.fig, self.width, self.height, self.rotate)
        pygame.surfarray.blit_array(self.surface, img)


def fig2rgb(fig, width, height, rotate):
    img1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = Image.frombuffer('RGB', fig.canvas.get_width_height(), img1)
    if rotate:
        img = img.resize((height, width), Image.BILINEAR)
    else:
        img = img.resize((width, height), Image.BILINEAR)
    img = np.array(img)
    if not rotate:
        img = img.transpose(1, 0, 2)
        img = img[:, ::-1, :]

    return img

