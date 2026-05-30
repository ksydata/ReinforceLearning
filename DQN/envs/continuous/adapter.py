from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np

from envs.base import EnvironmentAdapter


class ContinuousEnvironmentAdapter(EnvironmentAdapter):
    """CartPole 등 연속 상태 환경 어댑터 (자동 이산화)"""

    def __init__(self, env: gym.Env, bins: List[np.ndarray]):
        super().__init__(env)
        # 상속 클래스 EnvironmentAdapter의 __init__ 메서드 호출하여 학습환경 초기화
        self.bins = bins
        # 각 상태 차원별 bin 경계(np.linspace로 생성)를 받기 전 초기화

    def preprocess_state(self, state: Any) -> Tuple:
        # [구현] bins를 이용한 이산화(discretization) 로직
        s = np.asarray(state, dtype = float)
        # gym 환경에서 반환되는 상태값을 numpy 배열로 변환하여 변수에 저장
        discreted_indices = []
        # 각 상태 차원별로 이산화된 인덱스를 저장하기 위한 지수 리스트 초기화

        for i, b in enumerate(self.bins):
            # 각 상태 차원별로 bin 경계와 인덱스를 반복하여 처리
            discreted_index = int(np.digitize(s[i], b) - 1)
            # 상태값이 해당 bin 경계에 속하는 이산화된 인덱스 계산
            # [s] gym 환경에서 반환되는 상태값을 numpy 배열로 변환한 변수
            discreted_indices.append(discreted_index)
            # 계산된 이산화된 인덱스를 리스트에 추가

        return tuple(discreted_indices)
        # 이산화된 상태 차원별 인덱스 리스트를 튜플로 변환하여 반환
