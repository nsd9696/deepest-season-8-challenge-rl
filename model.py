import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

from utils import replay_loader

class AC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

class PPO:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if cfg.device=='cuda' and torch.cuda.is_available() else 'cpu')
        self.AC = AC(cfg.model.state_dim, cfg.model.action_dim, cfg.model.hidden_dim).to(self.device)
        optimizers = {
                      'Adam': optim.Adam,
                      'AdamW': optim.AdamW,
                      'Adagrad': optim.Adagrad,
                      'RMSprop': optim.RMSprop,
                      'SGD': optim.SGD
                      }
        self.optimizer = optimizers[cfg.optimizer.type](self.AC.parameters(), lr=cfg.optimizer.lr)
        self.gamma = cfg.model.gamma
        self.lambda_gae = cfg.model.lambda_gae
        self.epsilon_clip = cfg.model.epsilon_clip
        self.critic_coef = cfg.model.critic_coef
        self.entropy_coef = cfg.model.entropy_coef

    def get_action(self, state):
        policy, _ = self.AC(torch.Tensor(state).to(self.device))
        m = D.Bernoulli(policy.detach()[1])
        return int(m.sample().item())

    def update(self, memory, cfg):
        returns = torch.zeros(len(memory))
        advantages = torch.zeros(len(memory))
        states, actions, rewards, dones = memory.get()
        rewards = rewards.flatten()
        dones = dones.flatten()

        old_policies, old_values = self.AC(states.to(self.device))
        old_policies = old_policies.detach()
        
        running_return = 0
        previous_value = 0
        running_advantage = 0

        for t in reversed(range(len(memory))):
            running_return = rewards[t] + self.gamma * running_return * dones[t]
            running_tderror = rewards[t] + self.gamma * previous_value * dones[t] - old_values.view(-1).data[t]
            running_advantage = running_tderror + (self.gamma * self.lambda_gae) * running_advantage * dones[t]
            returns[t] = running_return
            previous_value = old_values.view(-1).data[t]
            advantages[t] = running_advantage

        for _ in range(cfg.n_epochs):
            for batch in replay_loader((states, actions, returns, advantages, old_policies), cfg.batch_size):
                batch_states, batch_actions, batch_retruns, batch_advantages, batch_old_policies = (x.to(self.device) for x in batch)

                policies, values = self.AC(batch_states)
                values = values.view(-1)
                
                ratios = ((policies / batch_old_policies) * batch_actions.detach()).sum(dim=1)
                clipped_ratios = torch.clamp(ratios, min=1.0-self.epsilon_clip, max=1.0+self.epsilon_clip)
                actor_loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages).sum()

                critic_loss = (batch_retruns - values).pow(2).sum()
                
                policy_entropy = (torch.log(policies) * policies).sum(0, keepdim=True).mean()

                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * policy_entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss