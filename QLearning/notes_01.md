# 1-1. 트이는 강화학습 [OT]


# 1-2. Q-Learning [맛집 찾기로 설명하는 강화학습]

* Greedy Action

4 by 4, 시작 지점 state space는 episode가 4가지이다. (위, 아래, 좌, 우)

episode 1. score = 1 (같은 점수일 경우, 랜덤하게 움직인다.)

맛집(상태공간)에 도달할 경우 받는 reward = 1이다. (episode의 점수를 reward로 준다.)

* Exploration

$$\epsilon$$-Greedy ($$0 <= \epsilon <= 1$$)

새로운 path, 새로운 맛집을 찾을 수 있다. 

reward에 $$\gamma$$를 곱한다. ($$0 <= \gamma <= 1$$)

* Q-update

$$Q(s_t, a_t) \leftarrow (1-a)*Q(s_t, a_t) + a(R_t + \gamma*maxQ(s_{t+1}, a_{t+1}))$$

$ \leftarrow $ 는 업데이트를 하라는 의미


$$a = 1$$이면, $$0 + 1*(R_t + \gamma*a_{t+1}))$$이면 그 다음값인 $$a_{t+1}$$은 없으므로, $$R_t$$이 된다. a가 새로운 걸 얼마나 받아들이는지. (soft copy)


# 2-1. Markov Decision Process [MDP] 

중요한 건 MDP는 PDF(이진(discrete)의 경우 Probability Mass Function)이 random하다는 것이다. expected return을 maximize하는 action하는 것이 goal이다.

$$P(a_1 \mid S_0, a_0, S_1)$$

$$P(a_1 \mid S_0, a_0, S_1, a_1)$$


# 2-2. 상태가치함수 V + 행동가치함수 Q + Optimal policy

* State value function 

지금 이순간부터 시작해서 기대되는 return, 현시점 t의 state에 대한 평가

이걸 optimize하는 게 optimal policy

$$Return G_t = R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} \cdots $$

$$E[x] = \int x \dot p(x) \,dx$$

(1) $$V(S_t) \triangleq \int _{a_t}^{\infty} G_t \cdot P(a_t, S_{t+1}, a_{t+1}, \cdots \mid S_t) /,d{a_t}^{\infty}$$ 조건부확률을 베이지안 정리로 풀어써보면 maximize하는 확률분포가 다 들어가있음

* Action value function

지금 state에서 행동으로부터 기대되는 return

(2) $$Q(S_t, a_t) \triangleq \int _{S_{t+1}}^{a_{\infty}} G_t \cdot P(S_{t+1}, a_{t+1},S_{t+2}, a_{t+2} \cdots \mid S_t, a_t) /,d{S_{t+1}}^{a_{\infty}}$$ 조건부확률


# 2-3. 벨만 방정식 [Bellman equation]

베이지안 정리 $$P(x,y) = P(x \mid y) \cdot p(y) \triangleq P(x,y \mid z) = P(x \mid y, z) \cdot p(y \mid z)$$ 

* State value function 

(2)-(1) = $$\int _{S_{t+1}}^{a_{\infty}} G_t \cdot P(S_{t+1}, a_{t+1},S_{t+2}, a_{t+2} \cdots \mid S_t, a_t) /,d{S_{t+1}}^{a_{\infty}} \cdot P(a_t \mid S_t /,da_t)$$

(1-1) = $$\int_{a_t} Q(S_t, a_t) \cdot P(a_t \mid S_t)$$

(1-2) $$\int_{a_t S_{t+1}} \int_{a_{t+1}:a_{infty}} (R_t + \gamma \cdot G_{t+1}P(a_{t+1}, \cdots \mid S_{t+1}) d_{a_{t+1}:a_{\infty}} P(a_t, S_{t+1} \mid S_t) /,d{a_t, S_t+1}$$

(1-2) $$\int_{a_t}\int_{S_{t+1}} \left[R_t + \gamma \cdot V(S_{t+1})\right] P(a_t, S_{t+1} \mid S_t) \,da_t\,dS_{t+1}$$ 헹동과 다음 상태를 동시에 확률로 모델링

(1-3) $$\int_{a_t}\int_{S_{t+1}} \left[R_t + \gamma \cdot V(S_{t+1})\right] P(S_{t+1} \mid S_t, a_t) \,da_t\,dS_{t+1}$$ 다음 상태 전이만 확률로 모델링(행동은 조건으로)

전이확률은 행동 후 다음 상태로 넘어갈 확률이며, 정책은 상태를 보고 행동을 결정하는 규칙을 말한다. 

* Action value function

Q라는 것을 action에 대해서 평균을 취한 것이 바로 V다. 

(2-1) $$P(a_{t+1} \mid S_t, a_t, S_{t+1}) \cdot P(a_t, S_{t+1} \mid s_T)$$

(2-2) $$\int_{S_{t+1}:a_{\infty}} P(a_{t+1}, S_{t+2}, a_{t+2}, \cdots \mid S_t, a_t, S_{t+1}) \cdot P(S_{t+1} \mid S_t, a_t)$$

(2-2) $$\int_{S_{t+1}:a_{\infty}} P(a_{t+1}, S_{t+2}, a_{t+2}, \cdots \mid S_{t+1}) \cdot P(S_{t+1} \mid S_t, a_t)$$

(2-3) $$\int_{S_{t+1}}\int_{a_t}^{\infty} \left[R_t + \gamma \cdot G_{t+1}\right] P(a_{t+1:\infty} \mid S_{t+1}) P(S_{t+1} \mid S_t, a_t) \,da_{t+1:\infty}\,dS_{t+1}$$

(2-3) $$\int_{S_{t+1}} \left[R_t + \gamma \cdot V(S_{t+1})\right] P(S_{t+1} \mid S_t, a_t) \,dS_{t+1}$$

(2-3) $$\int_{S_{t+1}, a_{t+1}}\int_{S_{t+2}, a_{\infty}} \left[R_t + \gamma \cdot G_{t+1} \cdot P(S_{t+2}:a_{\infty} \mid S_{t+1}, a_{t+1}) \right] P(S_{t+1}, a_{t+1} \mid S_t, a_t) \,dS_{t+1} \,d_{S_{t+2}:a_{\infty}}\,d_{S_{t+1}, a_{t+1}}$$

(2-3) $$Q(S_t, a_t) = \int_{S_{t+1}, a_{t+1}} \left[R_t + \gamma \cdot Q(S_{t+1}, a_{t+1})\right] P(S_{t+1}, a_{t+1} \mid S_t, a_t) \,dS_{t+1}\,da_{t+1}$$


# 3-1. Optimal policy [derivation]

(1) $$V(S_t) \triangleq \int _{a_t}^{\infty} G_t \cdot P(a_t:a_{\infty} \mid S_t) /,d{a_t}^{\infty}$$

$$= \argmax_{a_t} \int_{a_t} Q(S_t, a_t) \cdot P(a_t \mid S_t) \,da_t$$

$$P^*(a_t \mid S_t) = \delta(a_t - a_t^*)$$ 최적 정책은 최적 행동 $$at∗a_t^*
at∗​$$만 100% 선택한다. 