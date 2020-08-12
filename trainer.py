import gym
import torch

from model import PPO 
from utils import Replay


class PPOTrainer:
    def __init__(self, cfg):
        self.env = gym.make(cfg.env_name)
        self.env.seed(777)
        self.replay = Replay(cfg.replay)
        cfg.model['state_dim'] = self.env.observation_space.shape[0]
        cfg.model['action_dim'] = self.env.action_space.n
        self.ppo = PPO(cfg)

    def train(self, cfg):
        running_score = 0

        for i_episode in range(1, cfg.max_episodes+1):
            state = self.env.reset()
            score = 0

            for timestep in range(cfg.max_timesteps):
                action = self.ppo.get_action(state)
                state, reward, done, _ = self.env.step(action)
                done = 1 if done else 0
                score+=reward

                self.replay.push((state, [action], [reward], [done]))

                if done:
                    break

            self.ppo.update(self.replay, cfg)

            if i_episode % 10 == 0:
                print(f'episode : {i_episode} - score : {score}')

        self.replay.clear()
    
    def save(self, data_path, episode):
        raise NotImplementedError

    def load(self, data_path, episode):
        raise NotImplementedError

    def play(self, num_episodes=10):
        raise NotImplementedError

