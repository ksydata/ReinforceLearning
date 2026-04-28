# 5.1. Time Difference 기법 알고리즘 [현대 강화학습의 토대]

MC 몬테카를로 기법처럼 sample backup하되 Dynamic Programming과 같이 time-step마다 prediction할 수 있는 방식이다. 그렇기에 noise가 적어 학습의 variance가 낮지만 value fuction을 immediate reward만으로 bootstrapping하여(복원추출을 통해 true value를 estimate)하여 추정하므로 bias(편향)가 높다.

$e = $[-, s]a_0 [r_0, s_1]a_1 [r_1, s_2]a_2, \cdots [r_{t-1}, s]a [r, s']a'$

$v^{(n)}(s) \approx \frac{1}{n} \ \Sigma_{z in Z(s)} R(z), s in S$

시간차 유도 : n+1번째 궤적이 새로 발생 시 $+R(z_{n+1})$

$R(z)$는 종료 상태까지 가야 알 수 있다. (MC) 따라서 다음 state의 가치와 현 상태의 보상으로 대치하여 식을 유도한다. 

$v(s) = v(s) + p((r+\rho \cdot v(s')) - v(S))$

$r+\rho v(s') - v(s)$는 시간차 오차라고 부른다. 학습과정을 개념화한 식은 다음과 같다. 
> Esitimate_{new} = Estimate_{old} _ \Rho(Target - Esitimate_{old})$


# 5.2. TD 정책평가(Generalized Policy Improvement)

### 5.2.1. Sarsa [On-plicy]

state > action > reward > next state > next action

Sarsa를 이용한 정책 알고리즘의 핵심 (1) policy improvement
**상태 $s'$에서 $q$가 최대인 행동 $a'$를 결정한다.**

Sarsa를 이용한 정책 알고리즘의 핵심 (2) policy evaluation
> **$q(s,a) = q(s,a) + \Rho( (r+\rho \cdot q(s',a')) - q(s,a))$**
정책평가의 a'과 정책개선(업데이트) a'이 같다는 점에서 on-policy [?]
각 상태별로 선택되어지는 action이 [?]

epsilon-greedy policy control 방식 이용


### 5.2.2. Q-Learning [off-policy]

Sarsa를 개선한 알고리즘으로 딥러닝 또는 정책 그레디언트 방법과 결합하여 발전, 강화학습의 토대

$s, a, r, s'$으로 Sarsa에서 $a'$ 제거
> **$q(s,a) = q(s,a) + \Rho( (r+\rho \cdot max_{\hat{a}}q(s',\hat{a})) - q(s,a))$**

Q러닝의 오프 정책 성질 : 상태 s에서 q가 최대인 행동 $\hat{a}$(epsilon-greedy)와 $q$ 업데이트하는 행동 $a$(greedy)가 같을 수도 있고, 다를 수도 있다. 

즉, 현재 정책에서 벗어난 곳에 있는 정보를 가져다 쓰고, 이는 온 정책에 비해 샘플 간의 상관관계가 약해지므로 수렴에 도움이 된다. 



## 5.3. $\epsilon$ 스케줄링

$\epsilon$ 곱붕괴 / 합붕괴 스케줄링


## 5.4. Q-Learning 학습 곡선 


## 5.5. Q-Learning 블랙잭 게임 학습 

좋은 게임 전략을 구사하는 정책을 강화학습을 통해 알아낼 수 있는지





출처 : https://velog.io/@jameskoo0503/RLBasic-4/RLBasic-5