## Chapter 02. 강화학습 기초 다지기

1. 환경과 상호작용하는 에이전트

    https://untitledtblog.tistory.com/139

2. 마르코프 결정 프로세스(Markov Decision Process)

    2.1.1. decision에 의해 action이 생김. reward도 발생함

    value explaination of reinforcement learning 

    각 state마다 전체적인 reward를 최대화하는 action이 무엇인지를 결정하는 것

    $[-,s_0] \, a_0 \, [r_0, s_1] \, a_1 \, [r_1, s_2] \, a_2 \, \cdots \, [r_{t-1}, s_t] \, a_t \, [r_t, s_{t+1}] \, a_{t+1} \, \cdots \, [r_{T-2}, s_{T-1}] \, a_{T-1} \, [r_{T-1}, s_T]$

    첫 시작은 보상이 없기 때문에 상태 $s_0$에서 $r = -$

    $s$는 발생 가능한 모든 상태의 집합, $a$은 행동의 집합, $R$은 보상의 집합

    $E[\Sigma_{a=1}^T r_i]$ maximize

    $R(e)$는 episode, $R(z)$는 trajectory를 의미함


    2.1.2. 과업 사례

    이산 과업(discrete task), 행동이 이산값으로 표현

        ex1) state는 16개 공간, 1은 Frozen, 7은 Hole, 15는 Gole

        ex2) tick-tack-toe game

        ex3) CartPole는 x, x_velocity(속도), s는 실수(R)

        ex4) pong game
            
        ex5) Pendulnum, reward 함수의 원리는 알 수 없음

        ex6) BipedalWalker, reward 함수 {r=앞으로 전진한 거리-모터에 가한 힘의합, 넘어지면 -100}


    연속 과업(continuous task), 행동이 연속값으로 표현

        ex1) MuJoCo는 로봇 및 동물의 신체동작을 시뮬레이션하는 라이브러리


    2.1.3. 마르코프 결정 프로세스의 성질 (1)

    다음 순간의 상태 확률은 직전 순간만 고려한 상태 확률과 같다는 가정
    즉, 시간 t에서의 상태는 t-1에서의 상태에만 영향을 받는다

    $P(\hat{S} \mid S, a)$

    상태 전이 확률(state transition probability, MDP dynamics)    

    $P(\hat{S_{t+1}}, r_t \mid S_t, a_t)$

    stochastic task는 랜덤(frozen lake) $P(\hat{S} \mid S, a) < 1$

    determine task는 $P(\hat{S} \mid S, a) = 1$


    2.1.3. 마르코프 결정 프로세스의 성질 (2)

    모델 자유(model-free)는 상태와 보상 정보를 사용하여 학습하는 알고리즘 vs 모델 기반(model-based)

    모두 관찰하는 경우(fully-observable) vs 일부만 관찰하는 경우(POMDP)


    2.2. python, MDP programming
    
    2.2.1. Gymnasium library

    2.2.2. MDP algorithm


3. 랜덤 정책과 최적 정책의 기대 이익 비교

4. 정책과 가치함수

5. 강화학습의 난이도와 학습방법의 이해