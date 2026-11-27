"""Resumable collection of effective legacy-teacher actions with 64D observations.

Each completed episode commits an NPZ and a JSON summary. --out can be reused
after interruption; matching completed episodes are not simulated again.
Evaluation sets should be collected separately from the training problems.
"""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import time

import numpy as np

_WORK={}


def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as file:
        json.dump(data,file,ensure_ascii=False,indent=2)
        file.flush();os.fsync(file.fileno())
    tmp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def initialize(out):
    from inference.test import build_env,ROAD,EGO,TRAFFIC,SIMULATION,OBSERVATION,ACTION,REWARD
    from env.yeouido_env import make_env
    from env import road_builder
    import torch
    torch.set_num_threads(1)
    base=Path(__file__).resolve().parents[1]
    cfg=road_builder.build(ROAD,EGO,TRAFFIC,str(base/'env/sumo'))
    env=make_env(cfg_path=cfg,road=ROAD,ego=EGO,traffic=TRAFFIC,
                 mdp_sim=SIMULATION,mdp_obs=OBSERVATION,action=ACTION,
                 reward=REWARD,gui=False,teacher_assisted=True)
    assert env.observation_space.shape==(64,)
    assert env.lane_change_mode==257 and env.signal_assist and env.safety_shield
    assert not env.reroute and not env.route_lane_guidance
    Finalize(env,env.close,exitpriority=10)
    _WORK.update(env=env,out=Path(out))


def ego_state(env):
    from env.sumo_env import traci
    e=env.ego_id
    if e not in traci.vehicle.getIDList():return None
    return dict(speed=traci.vehicle.getSpeed(e),road=traci.vehicle.getRoadID(e),
                lane=traci.vehicle.getLaneIndex(e))


def one(problem):
    from env.sumo_env import traci
    from training.algorithms.sumo_expert import make_expert
    from training.assisted_labels import effective_action
    env=_WORK['env'];out=_WORK['out'];case=problem['id']
    options={k:problem[k] for k in ('level','origin','destination','snapshot_time','sumo_seed',
                                  'origin_at','destination_at') if k in problem}
    np.random.seed(problem['sumo_seed'])
    try:
        obs,_=env.reset(seed=problem['sumo_seed'],options=options)
    except RuntimeError as exc:
        result=dict(problem,ended_by='start_failed',length=0,error=str(exc),accepted=False)
        atomic_json(out/f'case_{case}.json',result)
        return result
    expert=make_expert(env,'legacy')
    assert not expert.manual_route_lanes
    origin=env.origin;destination=env.destination
    observations=[];next_observations=[];labels=[];requested=[];valid=[]
    terminated_buf=[];truncated_buf=[];arrived_buf=[];collision_buf=[];teleport_buf=[];red_buf=[]
    acceleration_raw=[];lane_before=[];lane_after=[]
    reasons=Counter();autonomous=0;control_changes=0;total_return=0.;info={}
    for step in range(env.max_steps):
        before=ego_state(env)
        action=expert.predict(obs)
        next_obs,reward,terminated,truncated,info=env.step(action)
        after=ego_state(env)
        label=effective_action(before,after,env.step_length,env.max_accel,env.max_decel)
        raw_teleport=(env.ego_id in traci.simulation.getStartingTeleportIDList()
                      or env.ego_id in traci.simulation.getEndingTeleportIDList())
        teleport=bool(info.get('teleported') or raw_teleport)
        if teleport:label['valid']=False;label['reason']='teleport'
        reasons[label['reason']]+=1
        if label['valid']:
            autonomous+=int(label['action'][1]!=0 and action[1]==0)
            control_changes+=int(not np.allclose(action,label['action'],atol=1e-4))
        observations.append(obs);next_observations.append(next_obs)
        labels.append(label['action']);requested.append(action);valid.append(label['valid'])
        acceleration_raw.append(label['unclipped_accel'] if label['unclipped_accel'] is not None else np.nan)
        lane_before.append(before['lane'] if before else -1);lane_after.append(after['lane'] if after else -1)
        terminated_buf.append(terminated);truncated_buf.append(truncated)
        arrived_buf.append(bool(info.get('arrived')));collision_buf.append(bool(info.get('collided')))
        teleport_buf.append(teleport);red_buf.append(bool(info.get('red_light')))
        obs=next_obs;total_return+=reward
        if terminated or truncated:break
    ended=('collision' if any(collision_buf) else 'teleport' if any(teleport_buf)
           else 'arrival' if info.get('arrived') else 'timeout' if truncated else 'disappeared')
    result=dict(problem,origin=origin,destination=destination,ended_by=ended,
                length=len(labels),valid_labels=sum(valid),label_reasons=dict(reasons),
                autonomous_lane_labels=autonomous,effective_control_changes=control_changes,
                red_lights=int(info.get('red_lights',sum(red_buf))),
                junction_overlaps=int(info.get('junction_overlaps',0)),
                return_sum=total_return,remaining=info.get('remaining'),
                congestion_removed_total=int(info.get('congestion_removed_total',0)),
                congestion_relief_active=bool(info.get('congestion_relief',{}).get('enabled',False)))
    result['lane_change_gate_blocks']=int(info.get('lane_change_gate_blocks',0))
    result['traffic_mode']=info.get('traffic_mode','uniform')
    if result['traffic_mode']=='fixed_population':
        result.update(fixed_population_target=info.get('fixed_population_target'),
                      fixed_population_created=info.get('fixed_population_created'),
                      fixed_population_max_excess=info.get('fixed_population_max_excess'))
    # Junction-overlap diagnostics are preserved; conservative training excludes
    # these episodes rather than treating disabled collision termination as proof.
    result['accepted']=(ended=='arrival' and result['red_lights']==0
                        and result['junction_overlaps']==0 and sum(valid)>0)
    tmp=out/f'case_{case}.npz.tmp'
    with tmp.open('wb') as file:
        np.savez_compressed(file,observation=np.asarray(observations,dtype=np.float32),
            next_observation=np.asarray(next_observations,dtype=np.float32),
            action_raw=np.asarray(labels,dtype=np.float32),
            action_requested=np.asarray(requested,dtype=np.float32),
            valid_label=np.asarray(valid,dtype=bool),
            effective_acceleration_unclipped=np.asarray(acceleration_raw,dtype=np.float32),
            lane_before=np.asarray(lane_before,dtype=np.int16),lane_after=np.asarray(lane_after,dtype=np.int16),
            episode=np.full(len(labels),case,dtype=np.int64),step=np.arange(len(labels),dtype=np.int32),
            terminated=np.asarray(terminated_buf,dtype=bool),truncated=np.asarray(truncated_buf,dtype=bool),
            arrived=np.asarray(arrived_buf,dtype=bool),collided=np.asarray(collision_buf,dtype=bool),
            teleported=np.asarray(teleport_buf,dtype=bool),red_light=np.asarray(red_buf,dtype=bool),
            level=np.full(len(labels),problem['level'],dtype='<U5'))
        file.flush();os.fsync(file.fileno())
    tmp.replace(out/f'case_{case}.npz')
    atomic_json(out/f'case_{case}.json',result)
    print(f"#{case} {problem['level']} {ended} steps={len(labels)} valid={sum(valid)} auto_lane={autonomous} red={result['red_lights']} overlaps={result['junction_overlaps']} clean={result['accepted']}",flush=True)
    return result


def summarize(records):
    return dict(problems=len(records),outcomes=dict(Counter(r['ended_by'] for r in records)),
        red_lights=sum(r.get('red_lights',0) for r in records),
        junction_overlaps=sum(r.get('junction_overlaps',0) for r in records),
        accepted_episodes=sum(r.get('accepted',False) for r in records),
        autonomous_lane_labels=sum(r.get('autonomous_lane_labels',0) for r in records),
        congestion_removed_total=sum(r.get('congestion_removed_total',0) for r in records),
        by_level={level:dict(problems=sum(r['level']==level for r in records),
            arrivals=sum(r['level']==level and r['ended_by']=='arrival' for r in records),
            accepted=sum(r['level']==level and r.get('accepted',False) for r in records))
                  for level in ('night','day','rush')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    # Import dependencies before creating workers to fail instead of respawning.
    from env.sumo_env import traci
    import torch
    if os.environ.get('LIBSUMO_AS_TRACI') and not traci.isLibsumo():
        raise RuntimeError('Requested libsumo is unavailable')
    problems=json.loads(args.problems.read_text())
    if len({p['id'] for p in problems})!=len(problems):raise ValueError('Duplicate episode ids')
    base=Path(__file__).resolve().parents[1]
    files=['training/collect_assisted_expert.py','training/assisted_labels.py',
           'training/algorithms/sumo_expert.py','env/driving_profile.py','env/yeouido_env.py',
           'env/sumo_env.py','env/road_config.py','env/congestion_relief.py']
    from env.road_config import ROAD
    from env.yeouido_env import load_places, place_name
    allowed=load_places('landmarks+gates')
    for problem in problems:
        place_name(allowed,problem['origin']);place_name(allowed,problem['destination'])
    population=None
    if os.environ.get('YEOUIDO_TRAFFIC_MODE')=='fixed_population':
        files+=['env/fixed_population.py','env/fixed_population_env.py']
        population=json.loads((base/'env/yeouido/fixed_population/manifest.json').read_text())['signature']
    manifest=dict(schema='assisted64_effective_actions_v1',problems=problems,
                  traffic_environment=dict(demand=ROAD.get('demand'),max_steps=ROAD.get('max_steps'),
                                           congestion_relief=ROAD.get('congestion_relief',{}),
                                           traffic_mode=os.environ.get('YEOUIDO_TRAFFIC_MODE','uniform'),
                                           fixed_population=population,
                                           effective_max_steps=int(os.environ.get('YEOUIDO_EPISODE_MAX_STEPS',ROAD['max_steps'])),
                                           min_trip=float(os.environ.get('YEOUIDO_MIN_TRIP',ROAD.get('min_trip',0)))),
                  source_hashes={p:hashlib.sha256((base/p).read_bytes()).hexdigest() for p in files})
    existing=args.out/'manifest.json'
    if existing.exists():
        if json.loads(existing.read_text())!=manifest:raise RuntimeError('Resume inputs/source differ from original collection')
    else:atomic_json(existing,manifest)
    records=[];pending=[]
    for problem in problems:
        p=args.out/f"case_{problem['id']}.json"
        try:
            record=json.loads(p.read_text())
            if record['ended_by']!='start_failed':
                with np.load(p.with_suffix('.npz'),allow_pickle=False) as data:
                    assert data['observation'].shape==(record['length'],64)
            records.append(record)
        except (FileNotFoundError,json.JSONDecodeError,ValueError,AssertionError,EOFError):pending.append(problem)
    started=time.monotonic()
    def save(complete):
        atomic_json(args.out/'results.json',dict(complete=complete,seconds=round(time.monotonic()-started),
            summary=summarize(records),records=sorted(records,key=lambda r:r['id'])))
    save(False)
    if pending:
        with multiprocessing.get_context('spawn').Pool(args.workers,initializer=initialize,initargs=(str(args.out),)) as pool:
            for record in pool.imap_unordered(one,pending,chunksize=1):
                records.append(record);save(False)
    save(True)
    print(json.dumps(summarize(records),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
