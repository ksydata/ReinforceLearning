# DPCore/gridDynamicProgramming.py

import numpy as np
from typing import List, Optional

class GridWorld:
    """4 by 4 Grid World Dynamic Programming의 핵심 로직 구현 클래스
    Args:
    - grid: 상태 가치를 저장하는 2차원 배열(rows x cols)
    - gamma: 할인율 또는 감쇠인자(0 <= gamma < 1)
    - terminal_states: 종료 상태 인덱스 목록(기본값은 마지막 칸)

    Logics:
    - update_once: 모든 진행 상태에서 벨만 백업을 수행하고 grid를 갱신
    - update_n(times = n): update_once를 times만큼 반복
    """
    def __init__(self, 
                 rows: int, cols: int, gamma: float, 
                 terminal_states: Optional[List[int]] = None):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros(
            shape = (rows, cols),
            dtype = float)
        self.gamma = gamma
        self.terminal_states = terminal_states if terminal_states is not None else [rows * cols - 1]
    
    def get_value(self):
        """주어진 그리드에서 상태 s의 벨만 방정식 백업 값을 계산"""
        return
    
    def update_once(self):
        """전체 상태에서 1회 벨만 백업을 수행"""
        return
    
    def update_n(self, times: int) -> np.ndarray:
        """update_once를 n회 반복하여 그리드 값 갱신"""
        return