import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Q‑Network
# 입력: 상태 벡터 (8차원)
# 출력: 각 행동에 대한 Q-value (4차원)
# 구조: 2개의 ReLU 히든층을 가진 다층 퍼셉트론 (MLP)
# 목적: Q(s, a) 값을 근사하기 위한 모델

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
# 확률적으로 무작위 선택 (탐험)
# 나머지는 네트워크가 가장 Q-value가 높은 행동 선택 (착취)
# epsilon은 학습 초반엔 높고, 점점 줄어듦 (탐험→착취로 전환)

def select_action(state, policy_net, epsilon, action_dim):    
    # 만약 랜덤(0.0~1.0) 이 e 보다 작으면, 탐험
    if random.random() < epsilon:
        return random.randrange(action_dim) 
    
    # 착취, 이 블록에서 역전파 비활성화
    with torch.no_grad():   
        # 벡터, torch tensor로 변환     
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        # unsqueeze -> state shape: [8] → [1, 8]
        # q_values -> Ex) tensor([[ 20.3,  18.7,  25.5,  10.1]]), 25.5 즉 메인엔진이 최선
        q_values = policy_net(state.unsqueeze(0))

        # torch.argmax(q_values, dim=1) -> dim=1 열, tensor([[ 20.3,  18.7,  25.5,  10.1]]) 중 tensor([2])
        # item() 함수로 인덱스 스칼라 추출, 2
        # 즉, action중 2, 메인엔진 점화를 리턴

        return int(torch.argmax(q_values, dim=1).item())  # 착취
    
# 3. 배치 학습
# 리플레이 버퍼에서 샘플을 추출하여, Q-learning 타겟에 따라 policy_net을 업데이트
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

    # policy_net(states) →
    # tensor([
    #   [ 5.1,  3.3,  7.8,  2.0],   # ← sample 1
    #   [ 1.0,  9.2, -0.5,  4.1],   # ← sample 2
    #   [ 6.3,  2.1,  3.0,  0.5],   # ← sample 3
    # ])

    #   actions = torch.tensor([[2], [1], [0]])  # shape: [3, 1]

    # q_values =
    # tensor([
    #   [7.8],   # ← Q(s1, a=2)
    #   [9.2],   # ← Q(s2, a=1)
    #   [6.3],   # ← Q(s3, a=0)
    # ])  # shape: [3, 1]
    
    # gather(1) 은 dim=1, 열을 모으는의미, policy_net 학습(순방향전파)
    q_values      = policy_net(states).gather(1, actions)  # Q(s,a)
    # q_values, 즉 Q(s,a) 벡터의 크기는 [batch_size, 1]

    # 현재의 Q값, q_values, 벨만 방정식의 좌변
    # 타켓의 Q값, next_q_max, 벨만방정식의 우변    
    with torch.no_grad():  
        next_q_max = target_net(next_states).max(1, keepdim=True)[0]
        target_q   = rewards + gamma * next_q_max * (1 - dones)

    # 둘의 차이 (q_values - target_q)^2 을 줄이는 것이 학습
    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 메인 루프
def main():
    env = gym.make("LunarLander-v3")                 # Box2D 환경 로드
    state_dim  = env.observation_space.shape[0]      # 8‑차원 상태
    action_dim = env.action_space.n                  # 4‑개 행동

    policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 하이퍼파라미터
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=50_000)
    batch_size      = 64
    gamma           = 0.99
    epsilon_start   = 1.0
    epsilon_end     = 0.05
    epsilon_decay   = 15000
    target_update   = 1000
    max_episodes    = 1000
    max_steps       = 1000

    global_step = 0
    scores = []

    # trange는 tqdm의 함수, range와 같지만 진행율 콘솔에 그림
    for ep in trange(max_episodes, desc="Episodes"):
        state, _ = env.reset(seed=ep)
        episode_reward = 0
        for _ in range(max_steps):
            # 엡실론(탐험률) 감소 e^0=1, e^-1 = 1/e = 0.3679, e^-2 = 1/e^2 = 0.1353 ...
            # 즉 초반엔 탐험
            epsilon = max(epsilon_end,
                          epsilon_start * np.exp(-global_step / epsilon_decay))    

            # 현재 상태에서 policy_net으로 Q값을 얻고, epsilon-greedy 방식으로 행동을 선택
            action = select_action(state, policy_net, epsilon, action_dim)

            # 선택한 행동을 진행 : 다음상태, 보상, 종료여부
            # gym에서 돌려주는 값 중,
            # terminated:   에이전트가 목표를 달성하거나 실패해서 게임이 끝난 경우
            # truncated:    시간 제한(또는 스텝 수 초과) 때문에 강제로 종료된 경우
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 경험리플레이 저장
            memory.append((state, action, reward, next_state, done))
            # 다음 상태로 이동            
            state = next_state
            # 현재 에피소드 보상 업데이트, 스텝증가
            episode_reward += reward
            global_step += 1

            # 경험리플레이가 batch_size보다 크면 Q-Learning으로 학습시작
            # policy_net만 학습
            train_step(memory, batch_size, policy_net,
                       target_net, optimizer, gamma)

            # 일정 주기마다 target_net을 업데이트하여 학습 안정화
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

    evaluate(policy_net, episodes=5)  # 5번 플레이 평가 시각화

def evaluate(policy_net, episodes=5):
    env = gym.make("LunarLander-v3", render_mode="human")  # 창 띄우는 모드
    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        total_reward = 0
        for _ in range(1000):
            action = select_action(state, policy_net, epsilon=0.0, action_dim=env.action_space.n)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"[평가 에피소드 {ep+1}] 보상: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()