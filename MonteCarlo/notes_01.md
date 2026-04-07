## 강화학습 4장. Monte Carlo

## 4.0. 개요 

Agent $leftrightarrow$ Environment

$\pi(s) \leftarrow action \\
s \rightarrow P(s' \mid s, \pi(s))$

몬테카를로 방법이란 MCP와 달리 확률분포 $P(s' \mid s, \pi(s))$ 
unknown(immediate, model-free)
에피소드($s_0$에서 $s_T$까지 가는 여정) 데이터를 이용하여 
정책 $\pi$ 또는 가치 함수 $V$를 개선하는 전략을 말한다. 

수학적으로 완전하고 흥미로운 관점도 많으며 성능도 괜찮으나 계산량이 좀 많고, 
에피소드 데이터를 무한하게 만들어야 한다는 점이 문제다. 
즉, **무언가 직접 측정하기 어려운 통계량이 있을 때, 여러 번 샘플링하여 그 값을 가늠(근사)하는 기법**을 말한다. $P_r(s' \mid s, a)$


## 4.1. 에피소드 데이터 수집 [ipynb 참고]

## 4.2. 탐험과 탐사의 균형

balance of between exploration(발견된 지점 집중 탐색) and exploitation(이곳저곳 고루 탐색), trade-off

## 4.2.1. 원리 (1) epsilon-greedy algorithm (가장 많이 사용함)

$a_t = \argmax_a(r + \gamma(v_{\pi}(s')))$
$u = [0,1]$ uniform distribution 사이의 난수(hard)
if $u < \epsilon, a_t$ 랜덤하게 행동을 선택(탐험)하거나 
else $\argmax_a(r + \gamma(v_{\pi}(s')))$하여 탐사

그안에서 변형을 해주면서 스케줄링이 일어나는데 향후 학습 예정

## 4.2.1. 원리 (2) epsilon-soft algorithm

state에서 action을 취할 확률값(soft)

$\pi(s_t, a) = \epsilon - \frac{\epsilon}{A(s_t)}$
아주 작은 값을 뺀다. 

## 4.2.1. 원리 (3) UCB(Upper Confidence Bound)

이전 이력을 고려한다. (MCTS)

## 4.2.2. 순수탐험과 엡실론-탐욕 알고리즘


## 4.2.3. 탐색 공간이 큰 경우의 분석





출처
- https://with-rl.tistory.com/entry/%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EB%B0%B0%EC%9A%B0%EB%8A%94-%EA%B0%95%ED%99%94-%ED%95%99%EC%8A%B5-05-MDP%EB%A5%BC-%EB%AA%A8%EB%93%A4-%EB%95%8C-%EB%B0%B8%EB%A5%98-%ED%8F%89%EA%B0%80%ED%95%98%EA%B8%B0
- https://with-rl.tistory.com/entry/%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EB%B0%B0%EC%9A%B0%EB%8A%94-%EA%B0%95%ED%99%94-%ED%95%99%EC%8A%B5-04-MDP%EB%A5%BC-%EC%95%8C-%EB%95%8C%EC%9D%98-%ED%94%8C%EB%9E%98%EB%8B%9D