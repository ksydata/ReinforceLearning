## 강화학습 4장. Monte Carlo [복습]

### 4.1. 몬테카를로 기법 Context

동적 계획법은 벨만 방정식을 통해 최적의 해를 찾아내는 방법으로서 마르코프 의사결정 프로세스(MDP)에 대한 모든 정보를 가진 상태에서 문제를 풀어가는 방식이다. 

단, 계산 복잡도가 state 크기 세제곱에 비례하므로 환경의 MDP를 모르더라도(model-free), 시행착오를 통해 환경과 상호작용하며 알아가는 방식으로 가치함수를 prediction하는 sample backup인 몬테카를로 기법을 활용한다.

이를 통해 정책을 update하는 방식으로 control한다. 즉, 에이전트는 가치함수를 최적화하여 최적 정책을 찾는다. 


### 4.2. MC의 원리

특히, MC는 terminal state까지 step별로 얻은 reward를 모두 저장하고, terminal state에서 state들의 가치함수(상태 또는 행동V)를 구하게 된다. 

$V_{\pi}(S) = E_{\pi}[G_t \mid S_t = s]$ (단, $G_t = R_{t+1} + \gamma \cdot R_{t+2} + \cdots + \gamma^{T-1} \cdot R_T$)

이 rewards 합의 평균(discounted, mutiple episodes를 마치고 얻은 returns의 기댓값)으로 가치함수를 근사한다. 

이 과정을 반복하여 true value function을 찾는다. 


### 4.3. Incremental Mean

$M_{k} = \frac{1}/{k} \Sigma_{j=1}^k x_j = \frac{1}/{k} (x_k + \Sigma_{j=1}^k x_j) = \frac{1}/{k} (x_{k} - M_{k-1}) = M_{k-1} + \frac{1}/{k} (x_k - M_{k-1})$

변형 시, $V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)} (G_t - V(S_t))$

(단, $\frac{1}{N(S_t)}$ = 1/에피소드 수행 횟수)

$V(S_t) \leftarrow V(S_t) + \alpha \cdot (G_t - V(S_t))$

$\alpha$는 편향성(오차, error), state의 중요도를 동일하게 취급하기 위해 값을 $\alpha$s로 고정한다. 

가치함수로 q-function의 식을 차용한다. 

---

### 4.4. MC로 상태가치함수 추정 및 정책 평가 시 버전(기준) 수정

에피소드 $e_i = [-,0] R[0,1] R[0,2] R[0,3] R[0,3] L[0,2] D[0,6] R[0,7]$

* 모든 방문(all-visit) 버전

Z(2) = [긴 궤적, 짧은 궤적] : 두 궤적 모두 포함

| 시간 | 상태 | 방문 |
|------|------|------|
| t = 0 | $s_0$ |  |
| t = 1 | $s_1$ |  |
| t = 2 | $s_2$ | 첫 방문 |
| t = 3 | $s_3$ | 첫 방문 |
| t = 4 | $s_3$ | 다음 방문 |
| t = 5 | $s_2$ | 다음 방문 |
| t = 6 | $s_6$ |  |
| t = 7 | $s_7$ | 종료(terminated) |


* 첫 방문(first-visit) 버전

Z(2)에서 긴 궤적인 첫 번째 궤적만 포함(같은 상태에서 시작하는 궤적이 둘 이상 발생했을 때, 첫 방문 버전으로 가장 긴 궤적만 취하는 몬테카를로 정책평가)

이득 계산 시 중복을 피하기 위해 에피소드를 역순으로 조회하는데, 그렇더라도 에피소드에서 해당 상태가 시간상 처음 나타난 시점에서 항상 더 긴 궤적만 취급한다. 
