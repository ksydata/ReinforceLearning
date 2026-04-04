## 강화학습 2장. Markov Process [Review]

2.1. 변수
* $\gamma$란 미래에 얻을 보상에 비해 당장 얻는 보상을 얼마나 더 중요하게 여길 것인지를 나타내는 0과 1 사이의 감쇠 인자를 말한다. $\gamma = 0$일 때는 현재의 보상만 생각하여 탐욕적(greedy)이며, $\gamma = 1$일 때는 현재와 미래의 보상이 동등하며 장기적인 시야를 갖는 보상을 의미한다. 
* 보상함수 $R = E[Rt \mid S_t = s]$란 어떤 상태 $s$에 도착했을 때 받게 되는 보상
* return은 $G_t = R_{t+1} \cdots \gamma^2 \cdot R_{t+3}$로 t시점부터 미래에 받을 감쇠된 보상의 합을 말하며, 강화학습의 목표는 리턴을 최대화하는 것이다.
* 에피소드는 $s_0, R_0, \cdots s_T, R_T$로 $s_0$에서 $s_T$까지 가는 여정을 말하며, 실제 확률분포를 알기 어려우므로 샘플링을 통해 어떤 값을 유추한다. 

2.2. 마르코프 성질
* 미래는 오로지 현재에 의해 결정되므로 이전 상태들은 영향을 주지 못한다. \ 
$P[S_{t+1} \mid S_t]$

2.3. 마르코프 결정 프로세스(MDP)
* 전이 확률 행렬 P란 MP, MRP에서 현재 상태가 $s$일 때 다음 상태가 $s'$가 확률 $P_{ss'}$을 말하며, MDP에서 현재 상태 $s$에서 agent가 행동 $a$를 선택했을 때 다음 상태가 $s'$이 될 확률을 말한다. \
$P_{ss'}{a} = P[S_{t+1} = s' \mid S_t = s, A_t = a]$
$R_{s}{a} = E[R_{t+1} \mid S_t = s, A_t = a]$

2.4. 정책함수[실행할 확률]와 가치함수(return의 기댓값)
* 정책함수란 agent 안에서 각 상태에서 어떤 행동을 선택할지 정해주는 함수로 더 큰 보상을 얻기 위해 교정해간다. \ 
$\pi(a \mid s) = P[A_t = a, S_t = s]$
* 상태가치함수란 agent의 정책 pi에 따라서 리턴 G가 달라지는 함수를 말한다. \
$u_{\pi}(s) = E_\pi[r_{t+1} + \gamma \cdot r_{t+2}\cdots \mid S_t = s] = E_{\pi}[G_t \mid S_t = s]$
* (state-)액션가치함수란 각 상태에서 액션에 대한 가치 평가 함수를 말한다. \
$q_\pi(s, a) = E[G_t \mid S_t = s]$

2.5. 최적정책 $\pi^*$ (optimal value function $v^*$)
* $\pi^*$와 $v^*$를 찾으면 MDP는 풀렸다고 말할 수 있다. 
* "The optimal value function specifies the best possible performance in the Markov Decision Process. An MDP is solved when we know the optimal value fn."



---



## 강화학습 3장. Dynamic Programming

3.0. 동적 계획법
* 큰 문제를 작은 부분으로 분해해서 문제를 풀고, 저장하여 재사용힌다. 
* 현재 상태의 가치 = 즉각 보상 + 미래 가치의 기대값(RL에서는 가치 함수 V(s), Q(s,a) 계산에 활용)

3.1. 벨만 (기대) 방정식과 재귀함수
* 벨만 방정식은 기본적으로 재귀적 관계에 대한 식으로 자기 자신을 호출하는 함수를 통해 현재 상태 $s_t$의 value인 return의 기댓값을 다음 step의 reward와 미래에 받을 reward를 더하는 방식으로 계산한다. 
* 1단계. $v_{\pi}(s) = \Sigma_{a in A} \pi(a \mid s) \cdot q_{\pi} (s \mid a)$ \
$q_{\pi}$를 이용하여 $v_{\pi}$를 계산한다. 이는 $s$의 가치는 $s$에서 $a$를 실행할 확률과 $s$에서 $a$를 실행하는 행동의 가치를 곱한 경우의 값들을 총합한 값이다. 
* 2단계. $q_{\pi}(s, a) = r_s^a + \gamma \cdot \Sigma_{s' in S} P_{ss'}^a \cdot v_{\pi}(s')$ \ 
$v_{\pi}$를 이용하여 $q_{\pi}$를 계산한다. 이는 $s$에서 $a$를 실행하는 행동의 가치를 $s$ 현재 상태에서 즉 시 얻는 보상과 $s$에서 $a$를 실행하면 다음 상태인 $s'$에 도착할 확률과 $s'$의 가치를 곱한 경우의 값들을 총합한 값이다. 

3.2. 벨만 방정식 내용 정리
* 1단계 $q_\pi$에 대한 식을 $v_\pi$에 대한 식에 대입하면 아래와 같음
$$v_\pi(s) = \sum_{a \in A} \pi(a|s) q_\pi(s,a) = \sum_{a \in A} \pi(a|s) \left( r_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_\pi(s') \right)$$

> 대입: $q_\pi(s,a) = r_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_\pi(s')$

* 반대로 $v_\pi$에 대한 식을 $q_\pi$에 대한 식에 대입하면 아래와 같음
$$q_\pi(s,a) = r_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_\pi(s') = r_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \sum_{a' \in A} \pi(a'|s') q_\pi(s',a')$$

> 대입: $v_\pi(s) = \sum_{a \in A} \pi(a|s) q_\pi(s,a)$

* 2단계 식을 계산하기 위해서 다음 2가지를 반드시 알아야 함
    - 보상 함수: $r_s^a$
    - 전이 확률: $P_{ss'}^a$

* $v{\pi}​(s)$: 상태 s가 얼마나 좋은지, "지금 내 위치가 얼마나 좋은가?"
* $q_{\pi}(s,a)$: 상태 s에서 행동 a를 했을 때 얼마나 좋은지, "지금 위치에서 오른쪽으로 가면 얼마나 좋은가?"

3.3. 벨만 최적 방정식
* 행동을 선택할 때 확률적으로 선택하는 것이 아니라 최댓값인 max 연산자를 통해 제일 좋은 행동을 선택한다. 따라서 행동은 확률(정책함수)을 따르지 않고 가치함수를 최대로 하는 행동을 선택한다. 