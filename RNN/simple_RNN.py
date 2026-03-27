import torch
import torch.nn as nn

# 1. 데이터 준비
chars = list('hello')
vocab = list(set(chars))
vocab_size = len(vocab)
embedding_dim = 10
print(vocab)

char2idx = {ch:i for i,ch in enumerate(vocab)}
idx2char = {i:ch for ch,i in char2idx.items()}
print(char2idx)

# 입력, 정답 만들기
x_data = [char2idx[c] for c in chars[:-1]]  # h e l l
y_data = [char2idx[c] for c in chars[1:]]   # e l l o
print(x_data)
print(y_data)

x = torch.tensor(x_data).unsqueeze(0)
y = torch.tensor(y_data)             
print(x)

# 2. 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super().__init__()
        hidden_size = 20

        # 문자수 만큼의 행, embedding_dim 만큼의 열을 갖는 word embedding weight vector 생성, 초기화
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN은 3차원 텐서를 입력받음 (batch, sequence, feature)
        # 위 embedding 통과후 batch=1, sequence=4, feature=10
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)          # (batch, seq, 10)
        out, _ = self.rnn(x)           # (batch, seq, 20)
        out = self.fc(out)             # (batch, seq, vocab)
        return out

model = SimpleRNN(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

# 3. 학습
model.train()
for i in range(epochs):            
    y_hat = model(x)    
    loss = criterion(y_hat.squeeze(0), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cost = loss.item()
    if i%10==0:
        print(f'epoch={i} cost={cost:.3f}')
    

# 4. 결과 확인
model.eval()
with torch.no_grad():
    y_pred = model(x)
    A = y_pred.argmax(dim=2)
    print(A)
    print([idx2char[i.item()] for i in A.squeeze()])