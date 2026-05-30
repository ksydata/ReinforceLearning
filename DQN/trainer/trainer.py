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

    def train(self, episodes: int, epsilon_decay: float, epsilon_min: float) -> Dict:
        """에이전트 학습 메서드

        Args:
            episodes: 학습할 에피소드 수
            epsilon_decay: 탐험확률 감소 비율 (0 < epsilon_decay < 1)
            epsilon_min: 최소 탐험확률 (0 <= epsilon_min < 1)
        
        Logics:
            - Є-탐욕: [0,1] 사이의 uniform distribution의 난수 u를 생성
            - if u < Є: 랜덤 선택, else(u >= Є): π로 선택
                * Є-곱 스케줄링: max(Є_min, ЄxЄ_decay) - Q-Learning
                * Є-합 스케줄링: max(Є_min, Є-Є_decay) - Q-Learning
            
        Returns:
            학습 과정에서의 보상 기록을 담은 딕셔너리
        """
        # [구현] 에피소드 루프 + epsilon 스케줄링


        return 
    
    def evaluate(self, episodes: int) -> Dict:
        """에이전트 평가 메서드

        Args:
            episodes: 평가할 에피소드 수
        
        Logics:
            탐험 없이 행동 선택 (탐사만)
        
        Returns:
            평가 과정에서의 보상 기록을 담은 딕셔너리
        """
        # [구현] 평가 루프
        # 탐험 없이 탐사만 해야 하므로 epsilon을 0으로 임시 설정 후 복원하는 방식 사용


        return
    
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
        # [구현] matplotlib 시각화
        # 에피소드 단위 보상은 노이즈가 크므로 이동 평균(rolling mean) 추가 시 추세 파악 용이


        
        return
