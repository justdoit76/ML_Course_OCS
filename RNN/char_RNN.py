import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 1. 데이터 준비 (여러 단어)
words = ['hello', 'world', 'thank', 'hey', 'machine_learning', 'love']
chars = sorted(list(set(''.join(words))))
print(chars)
char2idx = {ch: i + 1 for i, ch in enumerate(chars)} # 0은 패딩용으로 비워둠
char2idx['<PAD>'] = 0
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)

# 정수 인코딩 및 입출력 분리
x_data = []
y_data = []

for s in words:
    idxs = [char2idx[c] for c in s]
    x_data.append(torch.tensor(idxs[:-1])) # 입력: 'hell'
    y_data.append(torch.tensor(idxs[1:]))  # 출력: 'ello'
print(x_data)
print(y_data)

# 2. 패딩 처리, 문장 길이를 제일 긴 것에 맞춤
# batch_first=True로 설정하여 (Batch, Seq) 형태로 설정
x_padded = pad_sequence(x_data, batch_first=True, padding_value=0)
y_padded = pad_sequence(y_data, batch_first=True, padding_value=0)
print(x_padded)
print(y_padded)

print(f'x_padded={x_padded.shape}') # (Batch_size, Sequence_length)

# 3. 모델 정의
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

embedding_dim = 10
hidden_size = 20
model = CharRNN(vocab_size, embedding_dim, hidden_size)

# 패딩된 값(0)은 손실 계산에서 제외하도록 설정
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs=300

# 4. 학습 단계
for i in range(300):
    model.train()
    y_hat = model(x_padded)
    
    # Loss 계산을 위해 차원 변경: (batch_size*seq, C), 모든입력을 하나로 vector화
    # batch_size:6, seq:15
    loss = criterion(y_hat.view(-1, vocab_size), y_padded.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f'epoch={i} cost: {loss.item():.3f}')

# 5. 결과 확인
model.eval()
with torch.no_grad():
    predict = model(x_padded).argmax(dim=2)
    for j in range(len(words)):
        real_len = len(words[j]) - 1
        res = [idx2char[predict[j][k].item()] for k in range(real_len)]
        print(f'Input: {words[j][:-1]} -> Predict: {''.join(res)}')