from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import BaseAgent
from envs.base import EnvironmentAdapter


class AgentTrainer:
    """에이전트 강화학습 및 평가 통합 클래스

    Args:
        train(episodes): 에이전트 학습 시작
        evaluate(episodes): 평가
        plot_training_curve(rewards): 학습 곡선 시각화
    """
    def __init__(self, agent: BaseAgent, env_adapter: EnvironmentAdapter):
        self.agent = agent
        # BaseAgent 객체를 저장하는 인스턴스 변수
        self.env_adapter = env_adapter
        # AgentTrainer 클래스의 초기화 메서드

    def train(self, episodes: int, epsilon_decay: float, epsilon_min: float, decay_type: str = "multiplicative") -> Dict:
        """에이전트 학습 메서드

        Args:
            episodes: 학습할 에피소드 수
            epsilon_decay: 탐험확률 감소 비율 (0 < epsilon_decay < 1)
            epsilon_min: 최소 탐험확률 (0 <= epsilon_min < 1)
            decay_type: 탐험률 감소 방식 ("multiplicative" 또는 "additive")
        
        Logics:
            - Є-탐욕: [0,1] 사이의 uniform distribution의 난수 u를 생성
            - if u < Є: 랜덤 선택, else(u >= Є): π로 선택
                * Є-곱 스케줄링: max(Є_min, ЄxЄ_decay) - Q-Learning
                * Є-합 스케줄링: max(Є_min, Є-Є_decay) - Q-Learning
            
        Returns:
            학습 과정에서의 보상 기록을 담은 딕셔너리
        """
        self.epsilon_decay = epsilon_decay
        # 탐험률 감소 비율 저장 (epsilon_scheduling에서 참조)
        self.epsilon_min = epsilon_min
        # 탐험률 하한값 저장 (epsilon_scheduling에서 참조)

        rewards_history: List[float] = []
        # 학습 과정상 각 에피소드에서의 보상을 기록한 리스트 초기화
        epsilons_history: List[float] = []
        # 학습 과정상 각 에피소드에서의 탐험률을 기록한 리스트 초기화

        for t in range(episodes):
            state = self.env_adapter.reset()
            # 강화학습 환경 어댑터를 통해 환경 초기화된 상태
            action = self.agent.select_action(state)
            # 에이전트의 초기 행동 선택
            total_rewards = 0.0
            # 에피소드 단위 총 보상 초기화
            done = False
            # 에피소드 종료 여부 초기화

            while not done:
                next_state, reward, done, _ = self.env_adapter.step(action)
                # 환경에서 한 단계 진행
                next_action = self.agent.select_action(next_state)
                # 다음 상태에서의 행동 선택

                self.agent.learn(state, action, reward, next_state, next_action)
                # 에이전트 Q값 업데이트
                self.epsilon_scheduling(decay_type)
                # 탐험률 스케줄링 업데이트

                state = next_state  # 상태 업데이트
                action = next_action  # 행동 업데이트
                total_rewards += reward  # 에피소드 단위 보상 누적

            rewards_history.append(total_rewards)
            epsilons_history.append(self.agent.epsilon)
            # 학습 과정상 각 에피소드에서의 보상과 탐험률 기록

        return {"rewards_history": rewards_history,
                "epsilons_history": epsilons_history}

    def epsilon_scheduling(self, decay_type: str) -> float:
        """탐험률 스케줄링 메서드

        Logics:
            * Є-곱 스케줄링: max(Є_min, ЄxЄ_decay) - Q-Learning
            * Є-합 스케줄링: max(Є_min, Є-Є_decay) - Q-Learning
        """
        if decay_type == "multiplicative":
            # Є-곱 스케줄링
            self.agent.epsilon = max(
                self.epsilon_min,
                self.agent.epsilon * self.epsilon_decay)
        elif decay_type == "additive":
            # Є-합 스케줄링
            self.agent.epsilon = max(
                self.epsilon_min,
                self.agent.epsilon - self.epsilon_decay)
        return self.agent.epsilon
    
    def evaluate(self, episodes: int) -> Dict:
        """에이전트 평가 메서드 (환경별 맞춤형 평가)

        Args:
            episodes: 평가할 에피소드 수
        
        Logics:
            탐험 없이 행동 선택 (탐사만)
            - FrozenLake/Blackjack: 성공률 기반 평가 (보상 > 0인 비율)
            - CartPole: 평균 에피소드 길이 기반 평가 (solved 기준 195)
        
        Returns:
            평가 과정에서의 보상 기록을 담은 딕셔너리
        """
        saved_epsilon = self.agent.epsilon
        # 현재 탐험률 저장
        self.agent.epsilon = 0.0
        # 탐험 없이 탐사만 수행하도록 0으로 설정

        rewards_history: List[float] = []

        for _ in range(episodes):
            state = self.env_adapter.reset()
            total_rewards = 0.0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env_adapter.step(action)
                state = next_state
                total_rewards += reward

            rewards_history.append(total_rewards)

        self.agent.epsilon = saved_epsilon
        # 평가 후 탐험률 복원

        mean_reward = np.mean(rewards_history)
        
        # 환경 이름 확인 (env_spec.id 또는 env.spec.id)
        env_name = self.env_adapter.env.spec.id
        
        # 환경별 평가 지표 계산
        if "CartPole" in env_name:
            # CartPole: 에피소드 길이(보상)로 평가, gymnasium 기준 solved = 195
            solved_threshold = 195
            success_rate = np.mean([r >= solved_threshold for r in rewards_history])
            metric_name = "solved_rate"
            print(f"[{type(self.agent).__name__}] 평가 {episodes} 에피소드")
            print(f"  평균 에피소드 길이 : {mean_reward:.1f}")
            print(f"  Solved 비율 (≥{solved_threshold}) : {success_rate * 100:.1f}%")
        else:
            # FrozenLake, Blackjack: 보상이 양수인 비율로 평가
            success_rate = np.mean([r > 0 for r in rewards_history])
            metric_name = "success_rate"
            print(f"[{type(self.agent).__name__}] 평가 {episodes} 에피소드")
            print(f"  평균 보상 : {mean_reward:.3f}")
            print(f"  성공률   : {success_rate * 100:.1f}%")

        return {"rewards_history": rewards_history,
                "mean_reward": mean_reward,
                metric_name: success_rate,  # 환경별로 다른 키 이름 사용
                "success_rate": success_rate}  # 호환성을 위해 기존 키도 유지
    
    def plot_learning_curve(self, rewards: List[float]) -> None:
        """학습 곡선 시각화 메서드

        Args:
            rewards: 학습 과정에서의 보상 기록 리스트
        
        Logics:
            - x축: 에피소드 수, y축: 보상 값
            - 그래프 제목과 축 레이블 설정
            - 에피소드 단위 보상은 노이즈가 크므로 이동 평균(rolling mean) 추가 시 추세 파악 용이
        
        Returns:
            None
        """
        episodes_range = range(1, len(rewards) + 1)
        # x축: 에피소드 번호
        window = 100
        rewards_rollingmean = np.convolve(rewards, np.ones(window) / window, mode="valid")
        # 이동 평균(rolling mean) 계산 - 에피소드 단위 보상은 노이즈가 크므로 추세 파악 용이

        figure, ax = plt.subplots(figsize=(10, 5))
        # 그래프, 축의 크기를 설정한 객체 생성

        ax.plot(episodes_range, rewards,
                alpha=0.3, color="steelblue", label="Episode Reward")
        ax.plot(range(window, len(rewards) + 1), rewards_rollingmean,
                color="darkorange", linewidth=2, label=f"Rolling Mean (window={window})")

        ax.set_xlabel("Episode")  # x축 레이블 설정
        ax.set_ylabel("Total Reward")  # y축 레이블 설정
        ax.set_title(f"Learning Curve - {type(self.agent).__name__}")
        # 그래프 제목 설정
        ax.legend()  # 범례 추가
        plt.tight_layout()  # 레이아웃 조정
        plt.show()  # 그래프 시각화
