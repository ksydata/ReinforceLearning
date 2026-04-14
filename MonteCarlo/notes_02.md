## 강화학습 4장. Monte Carlo

### 4.3. 몬테카를로 방법으로 정책 평가

MC(Monte Carlo)도 정책 반복(General Policy Iteration) 전략에 따라 동작 
현대는 TD(Time Difference) 기법을 더 많이 활용

### 4.3.1. 정책 평가 알고리즘(Evaluation)

i. 상태 가치함수

확률분포(전이확률, 환경)를 모른다는 가정 $P(s' \mid s, a)$
$v_{\pi}(s) \approx \frac{1}{Z(s)} \ \Sigma_{z in Z(s)} R(z), s in S$
이득 계산에서 중복 피하기 위해 역순으로 상태 가치함수 추정 for 루프 반복

ii. 행동 가치함수

지나간 state 상태를 계속하여 돌아올 수 있음

### 4.4. 몬테카를로 방법으로 정책 학습(Improvement)

## 4.4.1. 학습 알고리즘

* 탐욕 선택 : $\epsilon-greedy$ 버전
* 탐험과 탐사 균형 : $\epsilon-soft$ 버전

### 4.5. 성능 향상 기법

TD(Time Difference) 시간차 기법, off policy에 form이 사용

### 4.4.1. 점증 계산법

$A_{t+1} = A_t + \frac{R_{t+1}}{A_t}$

### 4.4.2. 온 정책과 오프 정책

* On-policy [5장] : 각 state의 reward, action, state 간의 correlation 발생

* Off-policy(Target-policy) [7장] : Q-Learning, SARSA