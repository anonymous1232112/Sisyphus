import time
from copy import deepcopy

import numpy as np

from service.earlystop import measured_state
from .recovery import ExistingFollower,recover


def policy_loop(strategy, ctx, robot, events, features, fps, control_time_s,
                dataset, single_task, *, detector, control, finish, stop_on_empty=False,
                abort_detector,record_grasp,abort_home_state=None):
    from lerobot.rollout.strategies import episodic as stock

    interval = strategy._interpolator.get_control_interval(fps)
    started = time.perf_counter()
    timestamp = 0.
    reason = 'timeout'
    last_action = None
    terminal_recorded = False
    error = None
    recovery_report=None
    recovery_frame_template=None
    recovery_recording=dict(camera_read_calls=0,frames=0,image_source=None,
                            state_source='existing_follower_rpc')
    def recovery_frame(target,measured):
        nonlocal last_action
        control.check()
        state=measured_state({'observation.state':measured})
        elapsed=time.perf_counter()-started
        detector.update(state,elapsed)
        last_action=dict(zip([*[f'joint_{i}.pos' for i in range(6)],'gripper.pos'],map(float,target)))
        frame=dict(recovery_frame_template)
        frame['observation.state']=state.astype(np.asarray(frame['observation.state']).dtype,copy=True)
        action=stock.build_dataset_frame(features,last_action,prefix=stock.ACTION)
        dataset.add_frame({**frame,**action,'task':single_task})
        recovery_recording['frames']+=1
        recovery_recording['end_elapsed_s']=elapsed
    try:
        while timestamp < control_time_s:
            began = time.perf_counter()
            if events['exit_early'] or ctx.runtime.shutdown_event.is_set():
                events['exit_early'] = False
                reason = 'user_q'
                break
            raw = robot.get_observation()
            elapsed = time.perf_counter() - started
            if control.cancelled.is_set():
                reason = 'user_q'
                break
            done = detector.update(measured_state(raw), elapsed)
            check=abort_detector.update(measured_state(raw),elapsed,detector.last['tip_xyz_now'])
            if check is not None:
                record_grasp(check)
                if check['wrong_grasp_abort']:
                    strategy._engine.pause()
                    reason='wrong_grasp_abort'
                    control.check()
                    processed=ctx.processors.robot_observation_processor(raw)
                    recovery_frame_template=deepcopy(stock.build_dataset_frame(features,processed,prefix=stock.OBS_STR))
                    if 'observation.state' not in recovery_frame_template:
                        raise RuntimeError('Recovery recording requires the verified seven-value YAM state feature')
                    recovery_recording.update(image_source='cached_last_policy_observation',
                        start_elapsed_s=elapsed)
                    rpc=None
                    try:
                        rpc=ExistingFollower(robot,control,on_target=recovery_frame)
                        rpc.fk=detector.fk
                        recovery_report=recover(rpc,abort_home_state if abort_home_state is not None else detector.first_state)
                        control.check()
                    except Exception as exc:
                        recovery_report=dict(getattr(exc,'report',{}) or {})
                        recovery_report.update(ok=False,recovery_ok=False,
                            error=recovery_report.get('error') or type(exc).__name__+': '+str(exc),
                            failure_kind='user_q' if control.cancelled.is_set() else recovery_report.get('failure_kind','home_unverified'))
                        if recovery_report.get('final_state') is None:
                            cached=getattr(rpc,'last_state',None) if rpc is not None else None
                            recovery_report.update(final_state=list(cached if cached is not None else measured_state(raw)),
                                final_state_source='last_successful_follower_read' if cached is not None else 'last_policy_observation')
                    recovery_report['recovery_ok']=recovery_report.get('ok') is True
                    if not control.cancelled.is_set() and last_action is not None and recovery_report.get('final_state') is not None:
                        recovery_frame(list(last_action.values()),recovery_report['final_state'])
                        terminal_recorded=True
                    recovery_report['recording']=dict(recovery_recording)
                    if not recovery_report['recovery_ok']:
                        reason='user_q' if control.cancelled.is_set() else 'wrong_grasp_recovery_failed'
                        error=recovery_report.get('error') or 'Recovery failed'
                        control.fail_task(error,code=1)
                    break
            if elapsed >= control_time_s or done:
                reason = 'timeout' if elapsed >= control_time_s else 'early_done'
                if last_action is not None:
                    processed = ctx.processors.robot_observation_processor(raw)
                    frame = stock.build_dataset_frame(features, processed, prefix=stock.OBS_STR)
                    action = stock.build_dataset_frame(features, last_action, prefix=stock.ACTION)
                    dataset.add_frame({**frame, **action, 'task': single_task})
                    terminal_recorded = True
                break
            processed = strategy._process_observation_and_notify(ctx.processors, raw)
            if strategy._handle_warmup(ctx.runtime.cfg.use_torch_compile, began, interval):
                timestamp = time.perf_counter() - started
                continue
            action = stock.send_next_action(processed, raw, ctx, strategy._interpolator)
            if action is not None:
                action=getattr(getattr(robot,'inner',robot),'_resident3_last_action',action)
                event=abort_detector.note_action(action,measured_state(raw),time.perf_counter()-started)
                if event is not None:record_grasp(event)
                frame = stock.build_dataset_frame(features, processed, prefix=stock.OBS_STR)
                action_frame = stock.build_dataset_frame(features, action, prefix=stock.ACTION)
                dataset.add_frame({**frame, **action_frame, 'task': single_task})
                strategy._log_telemetry(processed, action, ctx.runtime)
                last_action = action
                if stop_on_empty and event is not None and event.get('event') == 'empty_grasp':
                    reason = 'empty_grasp'
                    terminal_recorded = True
                    break
            dt = time.perf_counter() - began
            if dt > interval:
                stock.logger.warning('Record loop is running slower (%.1f Hz) than target FPS (%s Hz).',
                                     1 / dt, fps)
            stock.precise_sleep(max(interval - dt, 0.))
            timestamp = time.perf_counter() - started
    except Exception as exc:
        if control.cancelled.is_set():
            reason = 'user_q'
        else:
            reason = 'wrong_grasp_recovery_failed' if abort_detector.report()['wrong_grasp_abort'] else 'error'
            error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        strategy._engine.pause()
        elapsed = time.perf_counter() - started
        finish(dict(event='execution_end', stop_reason=reason, elapsed_s=elapsed,
                    end_time_s=elapsed, wall_end_unix_s=time.time(),
                    max_seconds=float(control_time_s), conditions=detector.last,
                    terminal_observation_recorded=terminal_recorded, error=error,
                    abort_recovery=recovery_report,recovery_recording=recovery_recording,
                    recovery_ok=recovery_report.get('recovery_ok') if recovery_report else None,
                    settle_error_rad=recovery_report.get('settle_error_rad') if recovery_report else None,
                    tip_dist_m=recovery_report.get('tip_dist_m') if recovery_report else None,
                    recovery_failure_kind=recovery_report.get('failure_kind') if recovery_report else None,
                    **abort_detector.report()))
