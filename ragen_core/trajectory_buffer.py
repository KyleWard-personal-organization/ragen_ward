import numpy as np
from typing import List, Dict, Any

class TrajectoryBuffer:
    """
    RAGEN / StarPO 的轨迹回放池
    用于存储整条交互轨迹（包括多个状态、动作、奖励等），
    支持基于方差等指标进行过滤（Variance-based filtering），缓解 Echo Trap 问题。
    """
    def __init__(self):
        self.trajectories = []
        
    def add_trajectory(self, trajectory: List[Dict[str, Any]]):
        """
        添加一条轨迹。
        trajectory 是一个列表，列表每个元素是一个字典，记录单步的交互：
        {'state': str, 'action': str, 'reward': float, 'log_prob': float, ...}
        """
        self.trajectories.append(trajectory)
        
    def clear(self):
        self.trajectories = []
        
    def compute_returns(self, gamma: float = 1.0) -> List[float]:
        """
        计算每条轨迹的累积奖励
        """
        returns = []
        for traj in self.trajectories:
            total_r = sum([step['reward'] for step in traj])
            returns.append(total_r)
        return returns
        
    def filter_by_variance(self, group_size: int, retain_ratio: float = 0.25):
        """
        StarPO-S 核心改进：基于不确定性的轨迹过滤 (Variance-based filtering)
        假设同一个 prompt 我们采样了 group_size 次，计算这组轨迹的奖励方差。
        只保留方差最高的前 retain_ratio 比例的组（这些组对学习最有帮助）。
        """
        if not self.trajectories or group_size <= 1:
            return
            
        num_groups = len(self.trajectories) // group_size
        if num_groups == 0:
            return
            
        returns = self.compute_returns()
        variances = []
        
        for i in range(num_groups):
            group_returns = returns[i * group_size : (i + 1) * group_size]
            var = np.var(group_returns)
            variances.append(var)
            
        # 找出需要保留的组的索引
        num_retain = max(1, int(num_groups * retain_ratio))
        top_group_indices = np.argsort(variances)[-num_retain:]
        
        filtered_trajectories = []
        for idx in top_group_indices:
            filtered_trajectories.extend(
                self.trajectories[idx * group_size : (idx + 1) * group_size]
            )
            
        self.trajectories = filtered_trajectories
        
    def get_all_data(self):
        return self.trajectories
