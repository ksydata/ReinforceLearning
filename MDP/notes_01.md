## 마르코프 결정 프로세스(Markov Decision Process)

1. 이산과업

    Gymnasium library

    ```
    S(state) = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15} # 상태공간
    # 바둑에서의 상태공간|S|은 우주를 가득 채우는데 필요한 모래의 개수보다 훨씬 크다. 
    A(action) = {0(l), 1(d), 2(r), 3(u)} # 행동공간
    R(reward) = {1,0} # 보상공간

    4 by 4 array 
    isPlayer = Y, 1 
    isPlayer = N, 0
    ```

2. 연속과업

    Gymnasium library의 Pendulum 과업, MuJoCo library

3. MDP의 성질

    순간 0부터 순간 t까지 이력을 고려할 때 다음 순간 t+1의 상태확률은 직전 순간만 고려한 상태확률과 같다는 가정

    마르코프 성질에 따라 현재 순간 t의  상태 s_t에서 행동 a_t를 취했을 때, 다음 순간 t+1에서 상태 s_t+1과 보상 r_t가 발생할 확률분포

    즉, 전이 확률(transition probability) 또는 MDP 역학(MDP dynamics)이라고 한다. 이는 환경이 관장하는 정보로서 에이전트는 접근할 수 없으며, 상태와 보상 데이터를 통해서만 전이 확률을 간접적으로 경험한다. 이를 모델 자유라고 하며, 이와 달리 원칙을 깨고 전이 확률을 직접 활용하는 학습 알고리즘을 모델 기반이라고 한다. (동적 프로그래밍, 몬테카를로 트리 탐색)

    $$p(s_{t+1} \mid s_0, a_0, s_1, a_1, \cdots, s_t, a_t) = p(s_{t+1} \mid s_t, a_t)$$

4. 강화학습이 다루는 과업의 구분

    결정론 과업(deterministic task)

    스토캐스틱 과업(stochastic task) : 확률분포에 따라 난수(random num)를 생성

    핵심 목표는 기대 이득을 최대화하는 것


5. 관련 사이트

    매뉴얼 : https://gymnasium.farama.org

    소스코드 : https://github.com/farama-foundation/gymnasium