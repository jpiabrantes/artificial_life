import pickle
import os

import pygame

from replay.panes import BacteriaAttributeRenderer, PolicyRenderer, GameRenderer


HEIGHT = 512
WIDTH = 1024


class GlobalVariables:
    def __init__(self, dicts, env):
        self.iter = 0
        self.env = env
        self.dicts = dicts
        self.vision_mask = False
        self.tracking = False
        self.following_agent_id = 'agent_0'
        self.zoom = 24
        self.agent_names = None

    def update(self, iter_):
        self.iter = iter_
        dict_ = dicts[iter_]
        self.agent_names, _ = zip(*sorted([(k, int(k.split('_')[-1])) for k in dict_['agents']], key=lambda x: x[1]))
        if self.following_agent_id not in self.agent_names:
            self.following_agent_id = self.agent_names[-1]

    def iterate_agent(self, step):
        index = (step + self.agent_names.index(self.following_agent_id)) % len(self.agent_names)
        self.following_agent_id = self.agent_names[index]

    def scroll(self, step):
        self.zoom = max(min(self.zoom+step, 24), 2)


class MainHolder:
    def __init__(self, screen, g_variables):
        self.screen = screen
        screen.fill((255, 255, 255))
        self.width, self.height = self.screen.get_size()
        self.children = set()
        self.g_variables = g_variables
        self._set_layout()

    def _set_layout(self):
        game_screen = self.screen.subsurface(0, 0, self.width//2, self.width//2)
        self.children.add(GameRenderer(game_screen, self.g_variables))

        # attr_screen = self.screen.subsurface(self.width//2, 0, self.width//2, self.height//4)
        # self.children.add(BacteriaAttributeRenderer(attr_screen, g_struct))
        #
        # policy_screen = self.screen.subsurface(self.width // 2, self.height//4, self.width//2, self.height // 4)
        # self.children.add(PolicyRenderer(policy_screen, g_struct))

    def render(self, dict_):
        for child in self.children:
            child.render(dict_)


class GameController:
    def __init__(self, dicts, fps, main_holder, g_struct):
        self.dicts = dicts
        self.fps = fps
        self.main_holder = main_holder
        self.g_struct = g_struct
        self.pause = True
        self.iter = 0

    def _handle_keyboard_event(self, event):
        if event.key == pygame.K_SPACE:
            self.pause = not self.pause
        elif event.key == pygame.K_LEFT:
            if self.pause:
                self._iterate_step(-1)
        elif event.key == pygame.K_RIGHT:
            if self.pause:
                self._iterate_step(1)
        elif event.key == pygame.K_UP:
            self.g_struct.iterate_agent(1)
            self._iterate_step(0)
        elif event.key == pygame.K_DOWN:
            self.g_struct.iterate_agent(-1)
            self._iterate_step(0)
        elif event.key == pygame.K_t:
            self.g_struct.tracking = not self.g_struct.tracking
            self._iterate_step(0)
        elif event.key == pygame.K_v:
            self.g_struct.vision_mask = not self.g_struct.vision_mask
            self._iterate_step(0)

    def run(self):
        self._iterate_step(0)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    self._handle_keyboard_event(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_event(event)
            if not self.pause:
                self._iterate_step(1)
                clock.tick(fps)

    def _handle_mouse_event(self, event):
        if event.button == 4:
            self.g_struct.scroll(-1)
        elif event.button == 5:
            self.g_struct.scroll(1)
        self._iterate_step(0)

    def _iterate_step(self, step):
        self.iter = (self.iter + step) % len(self.dicts)
        self.g_struct.update(self.iter)
        main_holder.render(self.iter)
        pygame.display.flip()


clock = pygame.time.Clock()
fps = 24
if __name__ == '__main__':
    from envs.bacteria_colony.bacteria_colony import BacteriaColony
    from envs.bacteria_colony.env_config import env_default_config

    env = BacteriaColony(env_default_config)
    expname = 'EvolutionStrategies'

    with open(os.path.join('./dicts', expname+'.pkl'), 'rb') as f:
        dicts = pickle.load(f)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    g_struct = GlobalVariables(dicts, env)
    main_holder = MainHolder(screen, g_struct)
    game = GameController(dicts, fps, main_holder, g_struct)
    game.run()
