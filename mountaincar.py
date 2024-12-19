import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

# 定义超参数
ENV_NAME = 'MountainCar-v0'
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100  # 调整epsilon衰减速度
MEMORY_SIZE = 500000  # 增加经验回放缓冲区大小
BATCH_SIZE = 64
TARGET_UPDATE = 25  # 增加目标网络更新频率
MAX_STEPS = 200

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 增加网络宽度
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义测试函数
def test_model(model):
    model.to(device)
    env = gym.make(ENV_NAME, render_mode='human')
    state, _ = env.reset()
    state = torch.Tensor(state).unsqueeze(0).to(device)
    total_reward = 0
    for step in range(MAX_STEPS):
        q_values = model(state)
        action = torch.argmax(q_values).item()
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.Tensor(next_state).unsqueeze(0).to(device)
        total_reward += reward
        if done:
            break
        state = next_state
    print(f"Test Total Reward: {total_reward}")
    env.close()

# 定义主函数
def main():
    # 初始化环境
    env = gym.make(ENV_NAME)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    # 初始化DQN网络
    dqn = DQN(input_dim, output_dim).to(device)
    target_dqn = DQN(input_dim, output_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    # 初始化经验回放缓冲区
    memory = ReplayMemory(MEMORY_SIZE)
    # Epsilon衰减
    epsilon_by_episode = np.linspace(EPSILON_START, EPSILON_END, EPSILON_DECAY)
    # 初始化episode计数器
    episode = 0
    # 无限循环训练
    while True:
        state, _ = env.reset()
        state = torch.Tensor(state).unsqueeze(0).to(device)
        max_height = math.sin(3 * state[0][0])  # 初始化最高高度
        total_reward = 0
        for step in range(MAX_STEPS):
            epsilon = epsilon_by_episode[episode] if episode < EPSILON_DECAY else EPSILON_END
            # 选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = dqn(state)
                action = torch.argmax(q_values).item()
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.Tensor(next_state).unsqueeze(0).to(device)
            # 计算当前高度
            current_height = math.sin(3 * next_state[0][0])
            # 调整奖励
            if done:
                adjusted_reward = 5.0  # 达到终点的奖励
            elif current_height > max_height:
                adjusted_reward = reward + 2.0  # 达到更高高度的奖励
                max_height = current_height
            elif action == 2:  # 向右走
                adjusted_reward = reward + 1.0  # 增加向右走的奖励
            elif action == 0:  # 向左走
                adjusted_reward = reward + 1.0  # 增加向左走的奖励
            else:
                adjusted_reward = reward
            total_reward += adjusted_reward
            # 存储经验
            memory.push(state.cpu(), torch.LongTensor([action]), torch.Tensor([adjusted_reward]), next_state.cpu(), torch.Tensor([done]))
            # 更新状态
            state = next_state
            # 经验回放
            if len(memory) > BATCH_SIZE:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(BATCH_SIZE)
                batch_state = torch.cat(batch_state, 0).to(device)
                batch_action = torch.cat(batch_action, 0).to(device)
                batch_reward = torch.cat(batch_reward, 0).to(device)
                batch_next_state = torch.cat(batch_next_state, 0).to(device)
                batch_done = torch.cat(batch_done, 0).to(device)
                # 修复形状不匹配问题
                q_values = dqn(batch_state).gather(1, batch_action.unsqueeze(1))  # 确保形状为 [batch_size, 1]
                next_q_values = target_dqn(batch_next_state).detach().max(1)[0].unsqueeze(1)  # 确保形状为 [batch_size, 1]
                target = batch_reward + GAMMA * next_q_values * (1 - batch_done)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if done:
                break
        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        # 每100个episode后，测试并展示成果
        if episode % 100 == 0 and episode != 0:
            test_model(dqn)
        # 打印日志
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
        # 增加episode计数器
        episode += 1
    # 关闭环境
    env.close()

if __name__ == '__main__':
    main()