"""Recover executed teacher actions without confusing lane mappings with merges."""
import numpy as np


def effective_action(before, after, dt, max_accel, max_decel):
    """Label a complete SUMO step; return masks for ambiguous/unreachable steps.

    SUMO strategic lane changes occur during simulationStep, so the requested
    action is insufficient. Crossing into another road can renumber lanes and
    can hide a simultaneous merge: exclude those external-road boundary steps.
    Internal junction lanes do not permit lateral maneuvers in this task.
    """
    if dt <= 0 or max_accel <= 0 or max_decel <= 0:
        raise ValueError('Positive time step and acceleration limits are required')
    if not before or not after:
        return dict(action=np.zeros(2,dtype=np.float32),valid=False,
                    reason='ego_unavailable',unclipped_accel=None)
    acceleration=(after['speed']-before['speed'])/dt
    raw=acceleration/(max_accel if acceleration >= 0 else max_decel)
    valid=True;reason='ok'
    if before['road'].startswith(':'):
        lane=0
    elif before['road'] != after['road']:
        lane=0;valid=False;reason='external_road_boundary'
    else:
        lane=after['lane']-before['lane']
        if abs(lane)>1:
            valid=False;reason='multiple_lane_shift'
    if not np.isfinite(raw) or abs(raw)>1+1e-5:
        valid=False;reason='acceleration_outside_action_limits'
    return dict(action=np.asarray([np.clip(raw,-1,1),np.clip(lane,-1,1)],dtype=np.float32),
                valid=valid,reason=reason,unclipped_accel=float(raw))
