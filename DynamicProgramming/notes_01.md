## 3.2.1. 벨만 방정식 [교안 p.7]

벨만 방정식이라는 순환식을 유도하며, 이는 모든 강화학습 알고리즘의 근간을 형성한다. 

$v_\pi(s) = \Sigma_{a \in A(s)} \pi(a \mid s) \cdot q_\pi(s, a)$

$v_\pi(s) = \Sigma_{a \in A(s)} \cdot (r + \Sigma_{s`에서 출발하는 모든 궤적(trajactory)} p(z)R(z))$

z가 가진 가치는 딱 하나, $s`$ 은 다음 상태이고, 현재 상태 s에서 action을 가하면 다음 상태인 s`을 알고 있다고 가정한다.(결정론적 stocastic action을 취한다)

그렇다면, 이웃 상태의 값 $v_\pi(s`)$은 믿을만한 값일까? 설령 정확한 값이 아니고 학습 도중에 중간적인 값을 이용하더라도 **수렴한 가치 함수는 자기 일관성**을 유지한다. 이를 
bootstrapping 방식이라 한다.

## 3.2.2. 정책 평가(벨만 방정식 이용) 알고리즘

## 3.2.3. 정책 평가 실습 [주피터노트북 참고]

reward of action : left = 0, right = 2, down = 1, up = 3

## 3.2.4. 정책 반복 알고리즘

$v_{\pi_{1}}(s) \rightarrow \pi_2 \rightarrow v_{\pi_{2}}(s) \rightarrow \pi_3 \rightarrow \cdots$

GPI(generalized policy iteration) 알고리즘으로 설명, MDP모델(전이확률분포)로 평가한다. 3중 루프 형태로 계산량이 많다는 단점이 있다. 


## 3.3. 벨만 **최적** 방정식과 가치 반복 알고리즘 [교안 p.20] 

$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$

기댓값 연산을 최댓값을 취하도록 대체하여 계산량을 줄일 수 있다. 


## 3.4. 스토캐스틱 과업과 동적 프로그래밍(불확살성 반영)

스토캐스틱 환경을 위한 벨만 방정식과 벨만 최적 방정식
