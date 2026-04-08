import torch
import torch.nn as nn

# 1. 데이터 준비
chars = list('hello')
vocab = sorted(list(set(chars)))
vocab_size = len(vocab)
embedding_dim = 10
print(vocab)

char2idx = {ch:i for i,ch in enumerate(vocab)}
idx2char = {i:ch for ch,i in char2idx.items()}
print(char2idx)
print(idx2char)

# 입력, 정답 만들기
x_data = [char2idx[c] for c in chars[:-1]]  # h e l l
y_data = [char2idx[c] for c in chars[1:]]   # e l l o
print(x_data)
print(y_data)

x = torch.tensor([x_data])
y = torch.tensor([y_data])             
print(x)

# 2. 모델 정의
class SimpleLSTM(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super().__init__()
        hidden_size = 20

        # 문자수 만큼의 행, embedding_dim 만큼의 열을 갖는 word embedding weight vector 생성, 초기화
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # embedding_dim: 단어를 표현하는 벡터 차원
        # hidden_size: 지금까지 읽은 문맥을 저장하는 벡터 차원         
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):        
        x = self.embedding(x)           # in(1, 4), out(1, 4, 10)
        # 중요:RNN은 3차원 텐서를 입력받음 (batch, sequence, feature)        
        out, (h_n, c_n) = self.lstm(x)  # in(1, 4, 10), out(1, 4, 20), h_n:last hidden state, c_n:last cell state
        out = self.fc(out)              # in(1, 4, 20), out(1, 4, 4)
        return out

model = SimpleLSTM(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

# 3. 학습
model.train()
for i in range(epochs):            
    y_hat = model(x)    
    # y_hat(1, 4, 4), y(1,4)    
    # y_hat.view(-1, vocab_size)    => (4, 4) [[0.8, 0.1, 0.05, 0.05], [], [], []]
    # y.view(-1)                    => (4) [0, 2, 2, 3]
    loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))

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