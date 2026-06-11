import torch

class INS:
    def __init__(self,num_envs,dt,device):
        # 加载MuJoCo模型
        self.dt = dt
        self.num_envs = num_envs
        # 位姿
        self.Roll = torch.zeros(self.num_envs,1).to(device)
        self.Pitch = torch.zeros(self.num_envs,1).to(device)
        self.Yaw = torch.zeros(self.num_envs,1).to(device)
        self.YawAngleLast = torch.zeros(self.num_envs,1).to(device)  # 上一次偏航角
        self.Yaw_change_sum = torch.zeros(self.num_envs,1).to(device)  # 累积的偏航角变化
        self.update_count = torch.zeros(self.num_envs,1) .to(device) # 更新次数计数器
        self.YawRoundCount = torch.zeros(self.num_envs,1).to(device)  # 偏航角圈数计数器
        self.YawTotalAngle = torch.zeros(self.num_envs,1).to(device) # 偏航角总角度

        self.Yaw_change_rate = torch.zeros(self.num_envs,1).to(device)
        self.device = device

    def update(self, q):
        # 假设这是姿态更新逻辑（保持向量化）
        self.update_attitude(q)  # 确保此方法内部也是向量化实现
        
        # 计算偏航角变化 (向量化)
        yaw_change = self.Yaw - self.YawAngleLast
        
        # --- 处理偏航角圈数计数 (向量化) ---
        # 创建布尔掩码标识超过 ±π 的变化
        over_pi_mask = yaw_change > torch.pi
        under_neg_pi_mask = yaw_change < -torch.pi
        
        # 批量调整圈数计数器
        self.YawRoundCount[over_pi_mask] -= 1
        self.YawRoundCount[under_neg_pi_mask] += 1
        
        # --- 更新累积和计数器 (向量化) ---
        self.Yaw_change_sum += yaw_change
        self.update_count += 1
        
        # --- 每10步计算变化率 (向量化) ---
        # 创建掩码标识需要更新的环境
        update_mask = (self.update_count.squeeze(-1) >= 10)
        
        if update_mask.any():
            # 批量计算变化率
            self.Yaw_change_rate[update_mask] = self.Yaw_change_sum[update_mask] / (10.0 * self.dt)
            # 批量重置累积和与计数器
            self.Yaw_change_sum[update_mask] = 0.0
            self.update_count[update_mask] = 0
        
        # --- 计算总偏航角 (向量化) ---
        self.YawTotalAngle = 2 * torch.pi * self.YawRoundCount + self.Yaw
        
        # 更新上一次偏航角记录
        self.YawAngleLast = self.Yaw.clone()

    def reset(self,env_ids):
        self.Roll[env_ids] = 0
        self.Pitch[env_ids] = 0
        self.Yaw[env_ids] = 0
        self.YawAngleLast[env_ids] = 0
        self.Yaw_change_sum[env_ids] = 0
        self.update_count[env_ids] = 0
        self.YawRoundCount[env_ids] = 0 
        self.YawTotalAngle[env_ids] = 0
        self.Yaw_change_rate[env_ids] = 0

    def update_attitude(self,q):
        # 使用四元数计算欧拉角表示的姿态
        self.Pitch = torch.arcsin(2 * (q[:,:,0] * q[:,:,2] - q[:,:,3] * q[:,:,1]))
        self.Roll = torch.arctan2(2 * (q[:,:,0] * q[:,:,1] + q[:,:,2] * q[:,:,3]), 1 - 2 * (q[:,:,1]**2 + q[:,:,2]**2))
        self.Yaw = torch.arctan2(2 * (q[:,:,0] * q[:,:,3] + q[:,:,1] * q[:,:,2]), 1 - 2 * (q[:,:,2]**2 + q[:,:,3]**2))
        

