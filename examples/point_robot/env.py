import torch

class PointRobot:
    """二维质点接口样例。仅验证框架；不模拟腿式机器人。"""
    num_obs=4
    num_actions=2
    num_privileged_obs=None
    obs_history_length=1
    viewer=None
    dt=.02
    def __init__(self,cfg,sim_device,headless):
        self.cfg=cfg;self.device=sim_device;self.num_envs=1
        self.environment_config={}
        self.state=torch.zeros(1,4,device=sim_device)
    def reset(self):
        self.state.zero_();return self.state,None
    def get_observations(self):return self.state,None
    def step(self,a):
        self.state[:,2:]+=a.clamp(-1,1)*self.dt
        self.state[:,:2]+=self.state[:,2:]*self.dt
        reward=-(self.state[:,:2]-1).square().sum(-1)
        return self.state,None,reward,torch.zeros(1,dtype=torch.bool,device=self.device),{},None
    def close(self):pass
