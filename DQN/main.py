import argparse

import gymnasium as gym
# 강화학습 환경을 위한 라이브러리
import numpy as np
# 행렬 연산을 위한 라이브러리

from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent
# from agents.dqn import DQNAgent  # [구현 예정] DQN 등록
from envs.discrete import DiscreteEnvironmentAdapter
from envs.continuous import ContinuousEnvironmentAdapter
from trainer.trainer import AgentTrainer


# CartPole-v1 연속 상태 공간 이산화 bin 설정
CARTPOLE_BINS = [
    np.linspace(-4.8,  4.8,  9),  # 카트 위치
    np.linspace(-4.0,  4.0,  9),  # 카트 속도
    np.linspace(-0.42, 0.42, 9),  # 막대 각도
    np.linspace(-4.0,  4.0,  9),  # 막대 각속도
]

DISCRETE_ENVS  = {"FrozenLake-v1", "Blackjack-v1"}
# 이산 상태 환경 목록
CONTINUOUS_ENVS = {"CartPole-v1"}
# 연속 상태 환경 목록 (이산화 처리)

AGENT_MAP = {
    "sarsa"      : SarsaAgent,
    "q_learning" : QLearningAgent,
    # "dqn"      : DQNAgent,  # [구현 예정]
}
# 알고리즘 이름 → 에이전트 클래스 매핑


def build_adapter(env_id: str, env: gym.Env):
    """환경 ID에 맞는 어댑터를 생성하여 반환하는 함수"""
    if env_id in DISCRETE_ENVS:
        return DiscreteEnvironmentAdapter(env)
        # 이산 상태 환경 어댑터 반환
    if env_id in CONTINUOUS_ENVS:
        return ContinuousEnvironmentAdapter(env, CARTPOLE_BINS)
        # 연속 상태 환경 어댑터 반환 (bin 기반 이산화)
    raise ValueError(f"지원하지 않는 환경입니다 : {env_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",          type=str,   default="q_learning", choices=list(AGENT_MAP))
    parser.add_argument("--env",           type=str,   default="FrozenLake-v1")
    parser.add_argument("--episodes",      type=int,   default=2000)
    parser.add_argument("--rho",           type=float, default=0.1)
    # 학습률 α
    parser.add_argument("--gamma",         type=float, default=0.99)
    # 할인율 γ
    parser.add_argument("--epsilon",       type=float, default=1.0)
    # 초기 탐험률 ε
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    # ε 감소 비율 (곱 스케줄링)
    parser.add_argument("--epsilon-min",   type=float, default=0.01)
    # ε 하한값
    parser.add_argument("--decay-type",    type=str,   default="multiplicative", choices=["multiplicative", "additive"])
    # ε 감소 방식 (곱 스케줄링 / 합 스케줄링)
    parser.add_argument("--eval",          type=int,   default=20)
    # 평가 에피소드 수 (0이면 평가 스킵)
    args = parser.parse_args()

    env     = gym.make(args.env)
    # gymnasium 환경 생성
    adapter = build_adapter(args.env, env)
    # 환경에 맞는 어댑터 생성

    agent_cls = AGENT_MAP[args.algo]
    # 알고리즘 이름으로 에이전트 클래스 조회
    agent = agent_cls(
        action_space = env.action_space.n,
        rho          = args.rho,
        gamma        = args.gamma,
        epsilon      = args.epsilon,
    )
    # 에이전트 생성

    trainer = AgentTrainer(agent, adapter)
    # 학습/평가 통합 트레이너 생성

    result = trainer.train(
        episodes      = args.episodes,
        epsilon_decay = args.epsilon_decay,
        epsilon_min   = args.epsilon_min,
        decay_type    = args.decay_type,
    )
    # 학습 시작

    trainer.plot_learning_curve(result["rewards_history"])
    # 학습 곡선 시각화

    if args.eval > 0:
        eval_result = trainer.evaluate(episodes=args.eval)
        # 평가 시작 (eval=0이면 스킵)


if __name__ == "__main__":
    main()

