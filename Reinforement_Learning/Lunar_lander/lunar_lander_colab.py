# !pip install imageio
# !pip install swig
# !pip install "gymnasium[box2d]"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import imageio
from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Q‑Network
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# 2. ϵ‑greedy 행동 선택
def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)           # 탐험
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        q_values = policy_net(state.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())  # 착취
    
# 3. 배치 학습
def train_step(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

    states      = torch.tensor(states,      dtype=torch.float32, device=DEVICE)
    actions     = torch.tensor(actions,     dtype=torch.int64,   device=DEVICE).unsqueeze(1)
    rewards     = torch.tensor(rewards,     dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones       = torch.tensor(dones,       dtype=torch.float32, device=DEVICE).unsqueeze(1)

    q_values      = policy_net(states).gather(1, actions)  # Q(s,a)
    with torch.no_grad():
        next_q_max = target_net(next_states).max(1, keepdim=True)[0]
        target_q   = rewards + gamma * next_q_max * (1 - dones)

    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# 4. 메인 루프
def main():
    env = gym.make("LunarLander-v3")                 # Box2D 환경 로드
    state_dim  = env.observation_space.shape[0]      # 8‑차원 상태
    action_dim = env.action_space.n                  # 4‑개 행동

    policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=50_000)
    batch_size      = 64
    gamma           = 0.99
    epsilon_start   = 1.0
    epsilon_end     = 0.05
    epsilon_decay   = 15_000          # 스텝 단위 지수 감쇠
    target_update   = 1_000           # 스텝마다 타깃 네트워크 동기화
    max_episodes    = 1_000
    max_steps       = 1_000           # TimeLimit 기본값

    global_step = 0
    scores = []

    for ep in trange(max_episodes, desc="Episodes"):
        state, _ = env.reset(seed=ep)
        episode_reward = 0
        for _ in range(max_steps):
            epsilon = max(epsilon_end,
                          epsilon_start * np.exp(-global_step / epsilon_decay))
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            global_step += 1

            train_step(memory, batch_size, policy_net,
                       target_net, optimizer, gamma)

            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        scores.append(episode_reward)
        if (ep + 1) % 20 == 0:
            avg = np.mean(scores[-20:])
            print(f"[Episode {ep+1:4d}] 최근 20ep 평균 보상: {avg:7.2f}")

        # 간단한 해결 조건 – 평균 200 넘기면 종료
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 200:
            print("환경 해결!")
            break

    env.close()    
    evaluate(policy_net=policy_net)

def evaluate(policy_net, episodes=5):
    import imageio.v2 as imageio          # mimsave 쓰려면 v2 alias가 편함
    from IPython.display import Image, display

    env = gym.make("LunarLander-v3", render_mode="rgb_array")  # 창 띄우는 모드

    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        total_reward = 0

        frames = []

        for _ in range(1000):
            action = select_action(state, policy_net, epsilon=0.0, action_dim=env.action_space.n)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            frames.append(env.render())       # (H,W,3) uint8 배열

            if terminated or truncated:
                break
        print(f"[평가 에피소드 {ep+1}] 보상: {total_reward:.2f}")
        # GIF로 저장 (0.03 s ≈ 33 fps)
        
        imageio.mimsave("lander.gif", frames, duration=0.03)
        display(Image(filename="lander.gif"))     # Colab 셀에 즉시 표시
    env.close()    

if __name__ == "__main__":
    main()    