from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import gymnasium as gym


class EnvironmentAdapter(ABC):
    """강화학습 에이전트 환경 어댑터 클래스"""

    def __init__(self, env: gym.Env):
        """환경 어댑터 초기화"""
        self.env = env

    @abstractmethod
    def preprocess_state(self, state: Any) -> Any:
        """환경 상태를 에이전트가 사용할 형태로 변환"""
        pass

    def reset(self) -> Any:
        """환경 초기화 후 전처리된 상태 반환"""
        state, _ = self.env.reset()
        return self.preprocess_state(state)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """행동 수행 후 (다음상태, 보상, 종료여부, 정보) 반환"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # 환경에서 행동을 수행하여 다음 상태, 보상, 종료 여부, 정보 등을 변수에 저장 
        done = terminated or truncated
        # terminated : 에피소드가 목표 상태에 도달하거나 실패하여 종료된 경우
        # truncated : 에피소드가 최대 단계 수에 도달하여 종료된 경우
        return self.preprocess_state(next_state), float(reward), done, info

    def sample_action(self) -> int:
        """환경에서 무작위 행동을 샘플링하여 반환하는 메서드 (탐험)"""
        return self.env.action_space.sample()
        # method : def sample(mask: Any | None = None, probability: Any | None = None) -> Any

    @property
    def action_space_n(self) -> int:
        """행동 공간 크기"""
        return self.env.action_space.n
