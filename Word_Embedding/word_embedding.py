import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 입력 문장들
sentences = [
    'i love deep learning',
    'i like nlp',
    'i enjoy game',
    'deep learning is fun',
    'nlp is interesting'
]

# sentences = sentences*2
# print(sentences)

# 2. 단어 집합 만들기 (문장 분해)
words = set(' '.join(sentences).split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
# print(words)
# print(word2idx)
# print(idx2word)

vocab_size = len(words)
embedding_dim = 10
window_size = 2

# 3. skip-gram 데이터 생성 (중심 단어로 주변 단어를 예측)
# 3.1 'i love deep learning' 라면
# 3.2 'love' 가 중심단어라면 주변단어는 i, deep
# 3.3 입력 : love, 출력 : i, deep 이 나오는 구조
pairs = []
for sentence in sentences:
    tokens = sentence.split()
    for i, word in enumerate(tokens):
        left = max(0, i-window_size)
        right = min(len(tokens), i+window_size+1)
        for j in range(left, right):
            if i != j:
                pairs.append((word2idx[word], word2idx[tokens[j]]))
#print(pairs)

# 4. 모델 정의
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.output(x)
        return x

model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

# 5. 학습
for epoch in range(epochs):
    cost = 0
    for x, y in pairs:
        x = torch.tensor([x])
        y = torch.tensor([y])

        y_hat = model(x)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.item()

    if epoch % 10 == 0:
        print(f'epoch={epoch}, cost={cost:.3f}')


# 6. 임베딩 추출, Ex) W = vocab_size(5) * embedding_dim(10)
# 각 행이 하나의 단어
embeddings = model.embeddings.weight.detach().numpy()

# 7. PCA(Principal Component Analysis)로 2차원 축소
# 고차원 데이터의 특성을 유지하며 저차원으로 축소, 데이터의 본질적 구조 파악.
# 예를 들어 10차원이 조각상을 가장 잘 보이는 각도에서 사진촬영한 개념
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# 8. 시각화
plt.figure(figsize=(8,6))
for i, word in idx2word.items():
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.show()