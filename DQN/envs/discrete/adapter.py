from typing import Any

import numpy as np

from envs.base import EnvironmentAdapter


class DiscreteEnvironmentAdapter(EnvironmentAdapter):
    """FrozenLake, Blackjack 등 이산 상태 환경 어댑터"""

    def preprocess_state(self, state: Any) -> Any:
        # [구현] 이산 상태 환경은 변환 없이 상태를 그대로 반환
        if np.isscalar(state):
            # 상태가 스칼라(단일 값)인 경우
                return int( state )
                # 상태를 정수형으로 반환
        return state
        # 상태가 스칼라가 아닌 경우, 원 상태값 그대로 반환
