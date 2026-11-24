from dataclasses import dataclass
import math


FPS=30.
MAX_GRIPPER_STEP=.02


def normalized(value,name='gripper'):
    number=float(value)
    if not math.isfinite(number) or not 0<=number<=1:
        raise ValueError(f'{name} must be finite and in [0,1]')
    return number


@dataclass(frozen=True)
class OpenPlan:
    start: float
    target: float
    steps: int
    duration_s: float

    @classmethod
    def make(cls,start,target,seconds):
        start=normalized(start);target=normalized(target,'target')
        if not math.isfinite(seconds) or not 0<seconds<=30:
            raise ValueError('--seconds must be finite and in (0,30]')
        if target<=start:return cls(start,start,0,0.)
        steps=max(math.ceil(seconds*FPS),math.ceil((target-start)/MAX_GRIPPER_STEP))
        return cls(start,target,steps,steps/FPS)

    def at(self,step):
        if not 0<=step<=self.steps:raise ValueError('Step outside gripper plan')
        return self.start+(self.target-self.start)*step/self.steps if self.steps else self.start
