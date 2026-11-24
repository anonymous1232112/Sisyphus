from dataclasses import asdict,dataclass
import json

from robot.home import finite_vector
from .resident import ResidentTask
from .holding import RULE_VERSION,H_RULE_VERSION,HoldingSettings,settings_from_env as holding_from_env


def abort_settings(value):
    if not isinstance(value,dict) or set(value)!={'enabled','rule_version','holding','home_state'} or value['enabled'] is not True:
        raise ValueError('A2 requires explicit enabled abort_settings')
    if value['rule_version']!=RULE_VERSION:raise ValueError('Expected Sec.3.3 Eq.(8) abort protocol')
    holding=HoldingSettings.parse(value['holding']).report();home=value['home_state']
    if home is not None:
        home=list(finite_vector(home,(7,)))
        if not 0<=home[6]<=1:raise ValueError('Home gripper must be normalized')
    return dict(enabled=True,rule_version=RULE_VERSION,holding=holding,home_state=home)


@dataclass
class AbortTask(ResidentTask):
    abort_settings: dict | None = None
    abort_action_enabled: bool = True
    h_rule_version: str = H_RULE_VERSION
    relocate_on_empty: bool = False
    continuation: bool = False

    @classmethod
    def parse(cls,obj):
        if not isinstance(obj,dict) or 'abort_settings' not in obj:
            raise ValueError('This service only accepts explicit wrong-grasp-abort tasks')
        data=dict(obj);settings=abort_settings(data.pop('abort_settings'))
        action_enabled=data.pop('abort_action_enabled',True)
        if type(action_enabled) is not bool:
            raise ValueError('abort_action_enabled must be a boolean')
        if data.pop('h_rule_version',None)!=H_RULE_VERSION:raise ValueError('Expected '+H_RULE_VERSION)
        relocate=data.pop('relocate_on_empty',False)
        continuation=data.pop('continuation',False)
        if type(relocate) is not bool or type(continuation) is not bool:
            raise ValueError('relocate_on_empty and continuation must be booleans')
        if continuation and settings['home_state'] is None:
            raise ValueError('A continuation needs the home state')
        checked=ResidentTask.parse(data)
        return cls(**asdict(checked),abort_settings=settings,abort_action_enabled=action_enabled,
                   relocate_on_empty=relocate,continuation=continuation)


def settings_from_env(env):
    home=env.get('SIS_A2_HOME_STATE')
    return abort_settings(dict(enabled=True,rule_version=RULE_VERSION,holding=holding_from_env(env).report(),
                               home_state=json.loads(home) if home else None))
