"""Translate learned arrival coordinates between experimental workspaces.

Only units change: physical observations and desired physical arrivals are
preserved. This is used for old training references, never a tactical policy.
"""
import numpy as np
import torch
from airhockey.dynamics import workspace_in_sim


def bounds(options):
    b=workspace_in_sim(bounds_mm=options.get('workspace_bounds_mm'))
    return np.array([b[k] for k in ('min_x','max_x','min_y','max_y')])


def _array(value,like):
    return torch.as_tensor(value,dtype=like.dtype,device=like.device) if torch.is_tensor(like) else np.asarray(value,dtype=like.dtype)


def _copy(value):return value.clone() if torch.is_tensor(value) else value.copy()


def normalized_position(value,source,target):
    a,b=_array(bounds(source),value),_array(bounds(target),value)
    physical=a[[0,2]]+(value+1)/2*(a[[1,3]]-a[[0,2]])
    return 2*(physical-b[[0,2]])/(b[[1,3]]-b[[0,2]])-1


def action_coordinates(action,source,target):
    result=_copy(action)
    result[...,:2]=normalized_position(action[...,:2],source,target).clip(-1,1)
    cap=(.05+.95*((action[...,5]+1)/2)**2)*source.get('accel',60)/target.get('accel',60)
    result[...,5]=2*((cap-.05)/.95).clip(0,1)**.5-1
    return result


def observation_coordinates(obs,source,target,history):
    result=_copy(obs)
    frames=result[...,:42*history].reshape(-1,history,42)
    frames[...,31:33]=normalized_position(frames[...,31:33],source,target)
    frames[...,15:21]=action_coordinates(frames[...,15:21],source,target)
    # Observed previous targets may lie outside the older reference's box;
    # preserve that information instead of clipping the observation.
    original=obs[...,:42*history].reshape(-1,history,42)
    frames[...,15:17]=normalized_position(original[...,15:17],source,target)
    ratio=source.get('accel',60)/target.get('accel',60)
    frames[...,33]*=ratio
    frames[...,14]/=ratio
    return result


class CoordinateReference(torch.nn.Module):
    def __init__(self,net,source,target):
        super().__init__();self.net=net;self.source=source;self.target=target
        self.history,self.shot_conditioned,self.obs_dim=net.history,net.shot_conditioned,net.obs_dim

    def forward(self,obs):
        old=observation_coordinates(obs,self.target,self.source,self.history)
        mean,value=self.net(old)
        action=action_coordinates(mean.tanh(),self.source,self.target)
        return torch.atanh(action.clamp(-.999999,.999999)),value

    def action_mean(self,obs):
        old=observation_coordinates(obs,self.target,self.source,self.history)
        return action_coordinates(self.net.actor(self.net.trunk(old)).tanh(),self.source,self.target)

    def act(self,obs,stochastic=False):
        old=observation_coordinates(obs,self.target,self.source,self.history)
        return action_coordinates(self.net.act(old,stochastic=stochastic),self.source,self.target)

    def state_dict(self,*args,**kwargs):return self.net.state_dict(*args,**kwargs)
