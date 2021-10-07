# Efficient Estimation of Word Representations in Vector Space

태그: word2vec

### **Abstract**

- 큰 데이터 셋에서의 word 벡터화
- 적은 computational 비용으로 발전된 정확도 계산
- syntactic, semantic 관점의 유사도 계산
    
    

→ 기존의 단어들은 단어간의 거리(유클리디안 거리)를 이용해서 단어간의 유사도를 측정했는데, 

    사실 이 단어들은 가까이 있어도 여러 수준의 유사성을 가질 수 있음.

**구문 규칙 & 의미를 이용한 단어 표현**

vector('King') - vector('Man') + vector('Woman') ≈ Queen

구문 규칙을 이용한 벡터 연산의 결과는 어떤 단어와의 거리가 가까움(유사도가 큼)

+) 단어들이 벡터로 표현될 때 선형 규칙성을 보존

---

LSA(Latent Semantic Analysis), LDA(Latent Dirichlet Allocation)과 같이 기존의 연속적인 단어 표현을 추정하는 모델들이 있었음. LDA는 큰 데이터셋에 대해서 연산 비용이 많이 들어감.

→ 논문의 목표 : 단어들이 벡터로 표현될 때 선형 규칙성을 잘 보존하면서 (better than LSA), 복잡도를 낮추고 정확도를 올리는 것 

### 훈련 복잡도 (computational complexity)

$O=E \times T \times Q$

- E : number of training epochs
- T : number of words in the training set
- Q : defined further for each model architecture

### Feedforward Neural Net Language Model(NNLM)

복잡도  $Q=N\times D+N\times D \times H + H\times V$ 

- N : input 단어 개수 (N이 10일 때, projection layer (P)의 차원은 500~2000)
- D : 단어의 표현 차수
- H : hidden layer의 크기 (보통 500 ~ 1000)
- V : 단어의 크기

대부분의 복잡도는 $N\times D\times H$에 의해서 발생함. 

### Recurrent Neural Net Language Model(RNNLM)

기존의 NNLM을 이용하면 문맥의 길이에 한계점이 있기 때문에 이를 극복하기 위해서 이 모델을 제안. RNN은 시간적으로 연결된 hidden layer를 가지기 때문에 순환 모델이 short term memory를 가지게 됨. 이전 time stamp의 과거 정보가 hidden layer를 통해서 현재의 input으로 update가 됨.

RNN 모델의 시간복잡도

 $Q = H\times H + H\times V$

$H \times V$는 계층적 softmax를 사용하면 $H \times log_2(V)$로 효율적으로 축소됨. 대부분의 복잡도는 $H\times H$로 인해서 발생.

### Parallel Training of Neural Networks

DistBelief라는 분산 프레임워크 사용, NNLM feedforward 방식을 포함해서 이 논문에서 제안. 이 

병렬 training 에서는 Adagrad를 활용한 미니배니 비동기 경사하강법을 이용. 

## New Models

### Continuous Bag-of-Words Model(CBOW)

feedforward NNLM 구조랑 비슷한데, 비선형 hidden layer를 제거하고 projection layer가 모든 단어들과 공유됨. 단어의 순서가 projection에 영향을 끼치지 않는 이 구조를 bag-of-words 모델이라고 함. input으로 4개의 이전 단어와 이후의 단어로 각각 원핫벡터로 변환해서 들어가고 log-linear classifier를 만들었고(여기에서 윈도우 사이즈는 4) 다음에 나올 문제에서 가장 좋은 성능을 냄. 

훈련 복잡도는 $Q=N\times D + D \times log_2(V)$ 

CBOW모델은 bag-of-words와 다르게 문맥의 continuous distributed representation을 사용함. 

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled.png)

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled%201.png)

input layer에서 projection layer로 갈 때 가중치 행렬을 사용하는데, input의 원핫 벡터와 가중치 행렬이 곱해져서 생긴 결과로 벡터들이 projection layer에서 평균이 구해지게 됨. 그리고 이 평균 벡터는 두번째 가중치 행렬 W와 곱해지고 결과적으로 원핫 벡터들과 동일한 V 차원의 벡터가 나옴. 여기서는 4. 이 벡터에 대해서 softmax 함수를 적용하면 최종적으로 score vector가 나오게 됨. 이 값은 j 번째 단어가 중심 단어일 확률을 나타냄. cross entropy를 이용해서 오류를 측정. 

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled%202.png)

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled%203.png)

**→ 주변에 있는 단어들을 이용해서 중간에 있는 단어를 예측하는 방법** 

### Continuous Skip-gram Model

skip-gram은 CBOW와 비슷한데, 단어를 문맥을 통해서 예측하는 것이 아니라 같은 문장의 다른 단어에 대한 분류를 극대화함. 각 단어를 continuous projection layer가 있는 log-linear classifier의 input으로 사용함. 범위를 증가시키는 것이 결과 단어 벡터의 quality를 향상시켰지만 계산 복잡도가 올라감. 

계산 복잡도는 $Q=C\times (D+D\times log_2(V))$ (C는 단어의 최대 거리) 

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled%204.png)

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled%205.png)

여기서 C=5로 설정하고 R 범위는 1~C 에서 랜덤하게 선택함. 앞뒤로 R개의 단어를 정답 label로. 따라서 RX2 개의 단어들에 대한 분류를 해야함. 그리고 R+R개의 단어들을 output으로 함. 

**→ 중심 단어에서 주변 단어를 예측하는 것** 

### CBOW와 Skip-gram의 구조 비교

![Untitled](Efficient%20Estimation%20of%20Word%20Representations%20in%20Ve%208f6b9749d07842edbf079522060bc032/Untitled%206.png)

## Results

기존의 신경망 기반의 모델에 비해서 간단한 모델 구조를 사용해서 만든 단어 벡터가 좋은 성능을 가지는 것을 확인함. 구문론적(Syntactic), 의미론적(Semantic)인 단어 pair를 이용해서 테스트한 결과, 

- For Syntactic words : CBOW > Skip-gram > NNLM < RNNLM
- For Semantic words : Skip-gram > CBOW > NNLM < RNNLM
- Total : Skip-gram > CBOW > NNLM

다음과 같이 확인할 수 있었음.