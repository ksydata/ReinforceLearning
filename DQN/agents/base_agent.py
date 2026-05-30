from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict

import numpy as np


class BaseAgent(ABC):
    """강화학습 에이전트의 추상 클래스로 공통 인터페이스로 통합
    Args:
        모든 강화학습 에이전트는 이 클래스 상속
        구체 클래스에서 반드시 구현해야하는 메서드 
        - select_action() : 행동 선택 (탐험 vs 탐사)
        - learn() : Q값 업데이트 (알고리즘별로 다름)
    Returns:
        None
    """

    def __init__(self, action_space: int, rho: float, gamma: float, epsilon: float):
        """강화학습 에이전트의 초기화 메서드"""
        self.action_space = action_space 
        # 행동 공간의 크기
        self.rho = rho
        # 학습률 (0 < rho <= 1)
        self.gamma = gamma
        # 할인율 (0 <= gamma <= 1)
        self.epsilon = epsilon
        # 탐험률 (0 <= epsilon <= 1)

        # Q-테이블 초기화 (상태-행동 가치 함수 참조표)
        self.Q_table : Dict[Any, Dict[int, float]] = {}

    @abstractmethod
    def select_action(self, state: Any) -> int:
        """주어진 상태에서 행동을 선택하는 메서드
        Args:
            state: 현재 상태
        Returns:
            선택된 행동 (정수형)
        """
        pass
    
    @abstractmethod
    def learn(self, state: Any, action: int, reward: float, next_state: Any, **kwargs) -> None:
        """에이전트가 경험으로부터 학습하는 메서드
        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 보상
            next_state: 다음 상태
        Returns:
            None
        """
        pass

    def _ensure_state(self, state: Any) -> None:
        """Q-테이블에 state 키가 없으면 defaultdict로 초기화"""
        if state not in self.Q_table:
            # Q-테이블에 종료 상태가 아닐 경우
            self.Q_table[state] = defaultdict(float)
            # 행동 가치 힘수의 참조표를 저장하기 위한 딕셔너리를 defaultdict로 생성하여 변수 Q에 대입

    def get_Q_value(self, state: Any, action: int) -> float:
        """주어진 상태와 행동에 대한 Q값(에이전트가 받을 것으로 예상되는 총 보상의 미래 기대값)을 조회/반환하는 메서드
        """
        self._ensure_state(state)
        return self.Q_table[state][action]
        # Q-테이블에서 주어진 상태와 행동에 대한 Q값을 반환

    def set_Q_value(self, state: Any, action: int, value: float) -> None:
        """주어진 상태와 행동에 대한 Q값(에이전트가 받을 것으로 예상되는 총 보상의 미래 기대값)을 설정하는 메서드
        """
        self._ensure_state(state)
        self.Q_table[state][action] = value
        # Q-테이블에서 주어진 상태와 행동에 대한 Q값을 설정

    def get_max_Q_value(self, state: Any) -> float:
        """주어진 상태에서 가능한 행동들 중 최대가 되는 Q값을 조회/반환하는 메서드
        """
        if state not in self.Q_table or len(self.Q_table[state]) == 0:
            # Q-테이블이 종료 상태이거나 비어있는 경우
            return 0.0
            # 0으로 초기화

        return max(self.Q_table[state].values())
        # Q-테이블에서 주어진 상태에서 가능한 행동들 중 최대가 되는 Q값을 반환

    def get_better_action(self, state: Any) -> int:
        """랜덤 정책에서 주어진 상태에서 가능한 행동들 중 최대가 되는 행동을 조회/반환하는 메서드"""
        if state not in self.Q_table or len(self.Q_table[state]) == 0:
            # Q-테이블이 종료 상태이거나 비어있는 경우
            return np.random.randint(0, self.action_space)
            # 행동 공간에서 무작위로 행동 선택

        better_actions = [
            a for a, Q in self.Q_table[state].items()
            if Q == max( self.Q_table[state].values() ) 
        ] # Q-테이블에서 주어진 상태에서 가능한 행동들 중 Q값이 최대가 되는 행동을 리스트화

        return int(np.random.choice(better_actions))
        # Q값이 최대가 되는 행동들 중 무작위로 행동 선택
