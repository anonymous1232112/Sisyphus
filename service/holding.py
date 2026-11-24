from collections import deque
from dataclasses import asdict,dataclass,fields
import math

RULE_VERSION='sec3.3_eq8_v2'
H_RULE_VERSION='sec3.3_h_v1'


@dataclass(frozen=True)
class HoldingSettings:
    n_min: int = 5
    w_min: float = .12
    delta: float = .15
    m: int = 3
    eps: float = .001

    @classmethod
    def parse(cls,value):
        if isinstance(value,cls):value=asdict(value)
        if not isinstance(value,dict) or set(value)!={f.name for f in fields(cls)}:
            raise ValueError('Holding parameters must contain n_min,w_min,delta,m,eps')
        for key in ('n_min','m'):
            if type(value[key]) is not int or not 1<=value[key]<=120:
                raise ValueError(key+' must be an integer in [1,120]')
        numeric={}
        for key in ('w_min','delta','eps'):
            if isinstance(value[key],bool):raise ValueError('Invalid '+key)
            numeric[key]=float(value[key])
            if not math.isfinite(numeric[key]) or not 0<numeric[key]<1:
                raise ValueError(key+' must be finite and in (0,1), normalized gap units')
        return cls(n_min=value['n_min'],m=value['m'],**numeric)

    def report(self):return asdict(self)


def settings_from_env(env):
    values=HoldingSettings().report()
    for key in values:
        raw=env.get('SIS_A2_HOLDING_'+key.upper())
        if raw is not None:values[key]=int(raw) if key in ('n_min','m') else float(raw)
    return HoldingSettings.parse(values)


def gap(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Measured gap must be in [0,1]')
    return value


class HoldingTracker:
    def __init__(self,settings=None):
        self.settings=HoldingSettings.parse(settings or HoldingSettings())
        self.history=deque(maxlen=self.settings.m+1)
        self.closed_steps=0;self.command_closed=False;self.w0=None
        self.holding_seen=False;self.first_holding=None;self.last=None
        self.empty_grasp=False;self.empty_grasp_count=0;self.empty_events=[]
        self.last_elapsed=-1.;self.samples=0
        self.holding_now=False
        self.released_after_holding=False;self.release_count=0;self.first_release=None

    def observe(self,w,elapsed_s):
        w=gap(w);elapsed=float(elapsed_s)
        if not math.isfinite(elapsed) or elapsed<0 or elapsed<self.last_elapsed:raise ValueError('Nonmonotonic observation time')
        self.last_elapsed=elapsed;self.samples+=1;self.history.append(w)
        if self.command_closed:self.closed_steps+=1
        p=self.settings
        lag=self.history[0] if len(self.history)==p.m+1 else None
        inward=None if self.w0 is None else self.w0-w
        change=None if lag is None else abs(w-lag)
        conditions=dict(commanded_closed=self.command_closed and self.closed_steps>=p.n_min,
            nonzero_gap=w>=p.w_min,inward_motion=inward is not None and inward>=p.delta,
            stationary=change is not None and change<p.eps)
        self.last=dict(elapsed_s=elapsed,sample=self.samples,gap=w,w0=self.w0,
            closed_command_steps=self.closed_steps,gap_before_m=lag,inward_motion=inward,
            gap_change_m_steps=change,conditions=conditions,parameters=p.report())
        if all(conditions.values()):self.holding_now=True
        if not self.holding_seen and all(conditions.values()):
            self.holding_seen=True
            self.first_holding=dict(self.last,event='holding',holding=True)
            return dict(self.first_holding)
        return None

    def command(self,g,pre_command_w,elapsed_s):
        g=float(g)
        if not math.isfinite(g):raise ValueError('Command g must be finite')
        pre_command_w=gap(pre_command_w)
        closed=g<.5;event=None
        if closed:
            if not self.command_closed:self.w0=pre_command_w;self.closed_steps=0
        else:
            if self.command_closed and not self.holding_seen:
                self.empty_grasp=True;self.empty_grasp_count+=1
                event=dict(event='empty_grasp',empty_grasp=True,elapsed_s=float(elapsed_s),
                    w0=self.w0,gap=pre_command_w,command_g=g,closed_command_steps=self.closed_steps,
                    definition='reopen command after closed commands with no holding established')
                if len(self.empty_events)<32:self.empty_events.append(dict(event))
            if self.command_closed and self.holding_now:
                self.released_after_holding=True;self.release_count+=1
                if self.first_release is None:
                    self.first_release=dict(event='released_after_holding',elapsed_s=float(elapsed_s),
                        gap=pre_command_w,command_g=g)
                    event=dict(self.first_release)
            self.holding_now=False
            self.closed_steps=0;self.w0=None
        self.command_closed=closed
        return event

    def report(self):
        return dict(holding_parameters=self.settings.report(),holding_seen=self.holding_seen,
            first_holding=self.first_holding,holding_conditions=self.last,
            empty_grasp=self.empty_grasp,empty_grasp_count=self.empty_grasp_count,
            empty_grasp_events=list(self.empty_events),holding_now=self.holding_now,
            released_after_holding=self.released_after_holding,release_count=self.release_count,
            first_release=self.first_release)
