from typing import Any

import numpy as np

from agents.base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    """State-Action-Reward-State-Action Agent (On-policy TD learning)"""
    
    def select_action(self, state: Any) -> int:
        """상태 s에서 epsilon-탐욕 정책으로, 현재 정책을 따라 학습하며 행동 선택하는 메서드
        
        Logics:
            1. 탐험 : 랜덤 정책에서 난수가 epsilon(탐험확률)보다 작은 경우, 무작위 행동 선택
            2. 탐사 : 그렇지 않은 경우, Q값이 가장 높은 행동 선택
        """
        if np.random.random() < self.epsilon:
            # 랜덤 정책에서 난수가 탐험률보다 작을 조건에 해당하는 경우
            return np.random.randint(0, self.action_space)
            # 행동 공간에서 무작위로 행동 선택
        else :
            # 탐험률이 난수보다 크거나 같은 경우
            return self.get_better_action(state)
            # Q-테이블에서 Q값이 최대가 되는 행동들 중 무작위로 행동 선택
    
    def learn(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        **kwargs,
    ) -> None:
        """다음에 실제로 선택할 a'의 Q값을 사용하여 Q값 업데이트하는 메서드
        
        Logics:
            Sarsa 업데이트: Q(s,a) += α[r + γQ(s',a') - Q(s,a)]"""
        old_Q = self.get_Q_value( state, action )
        # 현재 상태와 현재 행동에 대한 Q값 조회
        next_Q = self.get_Q_value( next_state, next_action )
        # 다음 상태와 다음 행동에 대한 Q값 조회

        td_target = reward + self.gamma * next_Q
        # TD 타겟값 계산 (보상 + 할인된 다음 행동의 Q값)
        td_error = td_target - old_Q
        # TD 오차 (TD 타겟값과 현재 상태의 예측값(현재 Q값)의 차이)

        new_Q = old_Q + self.rho * td_error
        # Q값 업데이트 (현재 Q값 + 학습률을 고려한 TD 오차) - 현재 정책에 대한 의존 정책
        self.set_Q_value(state, action, new_Q)
        # 업데이트된 Q값을 Q-테이블에 저장
