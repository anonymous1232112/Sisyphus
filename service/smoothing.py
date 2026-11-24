from collections import deque
from dataclasses import asdict,dataclass
import math


@dataclass(frozen=True)
class SmoothSettings:
    mode: str = '0'
    window: int = 3
    alpha: float = .5

    @classmethod
    def parse(cls,obj):
        if not isinstance(obj,dict) or set(obj) != {'mode','window','alpha'}:
            raise ValueError('Invalid action smoothing settings')
        mode,window,alpha=obj['mode'],obj['window'],obj['alpha']
        if mode not in ('0','ma','ema') or type(window) is not int or not 1<=window<=31:
            raise ValueError('Smoothing mode 0/ma/ema; window integer 1..31')
        if type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<alpha<=1:
            raise ValueError('EMA alpha must be in (0,1]')
        return cls(mode,window,float(alpha))

    @classmethod
    def from_env(cls,env):
        mode=env.get('SIS_ACTION_SMOOTH','0')
        window=env.get('SIS_ACTION_SMOOTH_WINDOW','3')
        if mode=='1':mode='ma'
        elif mode.isascii() and mode.isdigit() and mode!='0':window=mode;mode='ma'
        return cls.parse(dict(mode=mode,window=int(window),alpha=float(env.get('SIS_ACTION_SMOOTH_ALPHA','.5'))))

    @property
    def lag_steps(self):
        return (self.window-1)/2 if self.mode=='ma' else (1-self.alpha)/self.alpha if self.mode=='ema' else 0.

    def report(self):
        return dict(asdict(self),low_frequency_group_delay_steps=self.lag_steps,
                    low_frequency_group_delay_s_at_30fps=self.lag_steps/30,
                    gripper_smoothed=False)


class ActionSmoother:
    def __init__(self,settings):
        self.settings=settings
        self.history=deque(maxlen=settings.window)
        self.previous=None

    def apply(self,target):
        values=[float(x) for x in target]
        if len(values)!=7 or not all(math.isfinite(x) for x in values):
            raise ValueError('Expected seven finite action targets')
        mode=self.settings.mode
        if mode=='ma':
            self.history.append(values[:6])
            joints=[math.fsum(row[i] for row in self.history)/len(self.history) for i in range(6)]
        elif mode=='ema' and self.previous is not None:
            a=self.settings.alpha
            joints=[a*x+(1-a)*old for x,old in zip(values[:6],self.previous)]
        else:joints=values[:6]
        self.previous=joints
        return joints+[values[6]]
