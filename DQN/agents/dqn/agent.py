from typing import Any

from agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent (Off-policy, Neural Network 기반)
    
    TODO:
        - network.py : PyTorch 신경망 구조 (Q-Network, Target Network)
        - replay_buffer.py : 경험 재플레이 메모리
        - select_action() : epsilon-탐욕 정책 (신경망 출력값 기반)
        - learn() : 미니배치 샘플링 → 손실 계산 → 역전파
    """

    def select_action(self, state: Any) -> int:
        # [구현 예정] 신경망 기반 epsilon-탐욕 행동 선택
        raise NotImplementedError

    def learn(self, state: Any, action: int, reward: float, next_state: Any, **kwargs) -> None:
        # [구현 예정] 리플레이 버퍼 저장 → 미니배치 샘플링 → TD 손실 역전파
        raise NotImplementedError
