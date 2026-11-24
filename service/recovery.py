import time

import numpy as np

from robot.home import HomePose,MotionLimits,drive_home,finite_vector,wait_result
from robot.gripper import OpenPlan

OPEN_TARGET=.98
OPEN_SECONDS=2.
OPEN_SETTLE_SECONDS=1.
OPEN_TOLERANCE=.02
HOME_TOL_M=.06


class RecoveryFailed(RuntimeError):
    def __init__(self,report):
        self.report=dict(report)
        super().__init__(report.get('error') or 'Wrong-grasp recovery did not complete')


def home_metrics(state,home_state,*,fk=None,home_tol=HOME_TOL_M):
    if not np.isfinite(home_tol) or not 0<float(home_tol)<=HOME_TOL_M:
        raise ValueError('Recovery home_tol must be in (0,0.06] metres')
    state=finite_vector(state,(7,));home_state=finite_vector(home_state,(7,))
    if fk is None:
        from bon.fk import YamFK
        fk=YamFK()
    tip=np.asarray(fk.tip(np.asarray(state[:6],float)),float)
    home_tip=np.asarray(fk.tip(np.asarray(home_state[:6],float)),float)
    if tip.shape!=(3,) or home_tip.shape!=(3,) or not np.isfinite(tip).all() or not np.isfinite(home_tip).all():
        raise ValueError('Invalid FK home measurement')
    dist=float(np.linalg.norm(tip-home_tip))
    return dict(tip_dist_m=dist,home_tol_m=float(home_tol),home_check_ok=dist<=home_tol,
        settle_error_rad=max(abs(a-b) for a,b in zip(state[:6],home_state[:6])),
        tip_xyz_now=tip.tolist(),tip_xyz_home=home_tip.tolist())


class ControlGate:
    def __init__(self,control):self.control=control;self.event=control.cancelled
    def check(self):self.control.check()


class ExistingFollower:
    def __init__(self,robot,control,*,on_target=None,limits=None):
        self.robot=getattr(robot,'inner',robot)
        self.arm=self.robot.arm
        self.control=control;self.gate=ControlGate(control)
        self.limits=limits or MotionLimits()
        self.on_target=on_target
        self.last_state=None

    def read(self):
        self.gate.check()
        obs=wait_result(self.arm._require().get_observations(),self.gate,self.limits.rpc_timeout)
        q=finite_vector(obs['joint_pos'],(6,7))[:6]
        grip=finite_vector(obs['gripper_pos'],(1,))[0]
        if not 0<=grip<=1:raise ValueError('Invalid measured follower gripper')
        self.gate.check()
        self.last_state=q+(grip,)
        return self.last_state

    def send(self,state):
        state=finite_vector(state,(7,))
        future=self.control.send_target(self.arm._require().command_joint_pos,np.asarray(state,float))
        wait_result(future,self.gate,self.limits.rpc_timeout)
        if self.on_target is not None:
            measured=self.last_state if self.last_state is not None else self.read()
            self.on_target(state,measured)


def recover(rpc,home_state,*,clock=time.monotonic,wait=None,output=print,fk=None,home_tol=HOME_TOL_M):
    began=clock();phase='opening';last_measured=None
    report=dict(ok=False,recovery_ok=False,opening_targets_sent=0,open_target=None,
        home=None,final_state=None,final_state_source=None,tip_dist_m=None,settle_error_rad=None,
        home_tol_m=float(home_tol),joint_settle_timed_out=False,error=None)
    fk=fk if fk is not None else getattr(rpc,'fk',None)
    try:
        limits=rpc.limits.validate();gate=rpc.gate;wait=wait or gate.event.wait
        report['limits']=vars(limits)
        initial=last_measured=finite_vector(rpc.read(),(7,));held=initial[:6]
        home=HomePose(tuple(finite_vector(home_state,(7,))),'A2 confirmed scene home / first measured frame')
        report.update(home_metrics(initial,home.state,fk=fk,home_tol=home_tol))
        target=max(initial[6],OPEN_TARGET)
        report['open_target']=target
        plan=OpenPlan.make(initial[6],target,OPEN_SECONDS)
        previous=clock()
        for step in range(1,plan.steps+1):
            gate.check();pause=1/limits.fps-(clock()-previous)
            if pause>0:wait(pause)
            gate.check();current=last_measured=rpc.read()
            if max(abs(a-b) for a,b in zip(current[:6],held))>limits.tracking_limit:
                raise RuntimeError('Abort gripper opening: follower tracking error exceeds go_home limit')
            previous=clock();rpc.send(held+(plan.at(step),))
            report['opening_targets_sent']+=1
        deadline=clock()+OPEN_SETTLE_SECONDS
        while True:
            gate.check();current=last_measured=rpc.read()
            if current[6]>=target-OPEN_TOLERANCE:break
            if clock()>=deadline:raise RuntimeError('Abort gripper failed to open; no automatic home motion')
            wait(1/limits.fps)
        phase='home_motion'
        current=last_measured=rpc.read()
        try:
            report['home']=drive_home(rpc,home,current,limits,gate,clock=clock,wait=wait,output=output)
        except RuntimeError as exc:
            if not str(exc).startswith('Home settling timed out;'):
                raise
            report.update(joint_settle_timed_out=True,joint_settle_warning=str(exc))
            report['home']=dict(stop_reason='joint_settle_timeout',warning=str(exc))
        phase='home_check'
        gate.check();final=last_measured=finite_vector(rpc.read(),(7,));gate.check()
        report.update(home_metrics(final,home.state,fk=fk,home_tol=home_tol),
                      final_state=list(final),final_state_source='fresh_follower_rpc_after_home')
        if final[6]<target-OPEN_TOLERANCE:
            phase='opening'
            raise RuntimeError('Abort gripper is no longer open after home motion')
        if not report['home_check_ok']:
            raise RuntimeError('Recovery FK home distance %.6f m exceeds %.6f m' % (report['tip_dist_m'],home_tol))
        report.update(ok=True,recovery_ok=True,elapsed_s=clock()-began,
            method='existing follower RPC; original OpenPlan + go_home quintic limits; final success by FK <= home_tol')
        return report
    except Exception as exc:
        cancelled=rpc.gate.event.is_set()
        cached=getattr(rpc,'last_state',None)
        if report['final_state'] is None and (cached is not None or last_measured is not None):
            report.update(final_state=list(cached if cached is not None else last_measured),
                          final_state_source='last_successful_follower_read')
            try:report.update(home_metrics(report['final_state'],home_state,fk=fk,home_tol=home_tol))
            except Exception:pass
        report.update(ok=False,recovery_ok=False,error=type(exc).__name__+': '+str(exc),
            failure_kind='user_q' if cancelled else 'gripper_open_failed' if phase=='opening' else
                'home_not_reached' if phase=='home_check' and report['tip_dist_m'] is not None else 'home_unverified',
            elapsed_s=clock()-began)
        raise RecoveryFailed(report) from exc
