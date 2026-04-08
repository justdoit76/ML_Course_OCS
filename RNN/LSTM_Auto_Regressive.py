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

    def forward(self, x, ht=None):        
        x = self.embedding(x)           # in(1, 4), out(1, 4, 10)
        # 중요:RNN은 3차원 텐서를 입력받음 (batch, sequence, feature)        
        out, hidden = self.lstm(x, ht)  # in(1, 4, 10), out(1, 4, 20), hidden(ht, ct)
        out = self.fc(out)              # in(1, 4, 20), out(1, 4, 4)
        return out, hidden

model = SimpleLSTM(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

# 3. 학습
model.train()
for i in range(epochs):            
    y_hat, _ = model(x)    
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
    # 시작 문자 설정
    start_char = 'h'
    result = start_char

    # 모델에 넣기 위해 텐서로 변환 (batch=1, seq=1)
    x_test = torch.tensor([[char2idx[start_char]]])
    # 초기값 h0
    hidden = None 

    for i in range(vocab_size):
        y_hat, hidden = model(x_test, hidden)

        # 가장 확률이 높은 글자의 인덱스 추출
        pred_idx = y_hat.argmax(dim=2).item()
        pred_char = idx2char[pred_idx]

        # 예측 글자 result에 더하기
        result += pred_char
        
        # 중요: 방금 예측한 글자를 다음 단계의 입력으로 사용
        x_test = torch.tensor([[pred_idx]])

    print(f'pred={result}')