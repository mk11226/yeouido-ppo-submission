"""Joint lane/speed expert for the unassisted, fixed-route SUMO task.

All control is expressed as [normalized acceleration, adjacent lane command].
No route changes, teleports, vehicle moves, or environment assistance are used.
"""
from dataclasses import dataclass
import math

import numpy as np
import traci
import traci.constants as tc

from training.algorithms.sumo_expert import SumoExpert


@dataclass(frozen=True)
class ExpertSettings:
    acceleration: float = 2.0
    comfortable_brake: float = 3.0
    front_buffer: float = 2.0
    speed_buffer_time: float = 0.35
    signal_buffer: float = 3.0
    lane_buffer: float = 1.5
    lane_time_gap: float = 0.5
    jerk_limit: float = 2.5
    minimum_merge_room: float = 7.0
    spawn_guard: float = 8.0


def stopping_speed(gap, brake, dt, margin=0.0, leader_speed=0.0, leader_brake=4.5):
    """One commanded step followed by braking fits the available stopping room."""
    if brake <= 0 or dt <= 0 or leader_brake <= 0:
        raise ValueError('Braking limits and step length must be positive')
    available=max(0.0,float(gap)-margin)+max(0.0,float(leader_speed))**2/(2*leader_brake)
    return max(0.0,-brake*dt+math.sqrt((brake*dt)**2+2*brake*available))


def project_position(point,shape):
    """Project the ego front onto a neighboring lane's longitudinal coordinate."""
    points=np.asarray(shape,dtype=float)
    if len(points)<2:
        return 0.0
    segments=points[1:]-points[:-1]
    lengths=np.linalg.norm(segments,axis=1)
    fractions=np.clip(np.sum((np.asarray(point)-points[:-1])*segments,axis=1)/np.maximum(lengths**2,1e-12),0,1)
    projections=points[:-1]+fractions[:,None]*segments
    index=int(np.argmin(np.sum((projections-np.asarray(point))**2,axis=1)))
    return float(lengths[:index].sum()+fractions[index]*lengths[index])


def minimum_stopping_distance(speed,brake,dt):
    """SUMO's semi-implicit step: brake first, then travel at the new speed."""
    if brake<=0 or dt<=0:
        raise ValueError('Braking limit and step length must be positive')
    n=int(max(0,speed)//(brake*dt))
    return max(0.0,dt*(n*speed-brake*dt*n*(n+1)/2))


class RobustExpert(SumoExpert):
    """Explicit strategic lane selection with joint longitudinal safety bounds."""
    def __init__(self,env,settings=None):
        super().__init__(env,lookahead=150.0,accel=2.0)
        self.settings=settings or ExpertSettings()
        self._decision_key=None
        self._last_action=None
        self._last_acceleration=0.0
        self._episode=None
        self.last_decision={}
        self._shape_cache={}
        self._yellow_modes={}
        self._step_time=0.0
        self._previous_speed=None

    def target_speed(self,accel_limited=True):
        self.predict()
        if not self.last_decision:
            return 0.0
        return (self.last_decision['reachable_speed'] if accel_limited
                else min(self.last_decision['bounds'].values()))

    def lane_change(self):
        return int(self.predict()[1])

    def _next_edge(self):
        route=traci.vehicle.getRoute(self.ego)
        index=traci.vehicle.getRouteIndex(self.ego)
        return route[index+1] if 0<=index<len(route)-1 else None

    def _links(self,lane,next_edge):
        try:
            links=traci.lane.getLinks(lane,extended=True)
        except TypeError:
            links=traci.lane.getLinks(lane)  # libsumo always returns extended links
        if next_edge is None:
            return []
        return [link for link in links if link[0].rsplit('_',1)[0]==next_edge]

    def _neighbors(self,target_index):
        e=self.ego
        current=traci.vehicle.getLaneIndex(e)
        direction=1 if target_index>current else -1
        edge=traci.vehicle.getRoadID(e)
        target=f'{edge}_{target_index}'
        leaders=list(traci.vehicle.getLeftLeaders(e) if direction>0 else traci.vehicle.getRightLeaders(e))
        followers=list(traci.vehicle.getLeftFollowers(e) if direction>0 else traci.vehicle.getRightFollowers(e))
        if target not in self._shape_cache:
            self._shape_cache[target]=traci.lane.getShape(target)
        position=project_position(traci.vehicle.getPosition(e),self._shape_cache[target])
        my_gap=traci.vehicle.getMinGap(e)
        my_length=traci.vehicle.getLength(e)
        for vehicle in traci.lane.getLastStepVehicleIDs(target):
            if vehicle==e:
                continue
            other_pos=traci.vehicle.getLanePosition(vehicle)
            if other_pos>=position:
                gap=other_pos-traci.vehicle.getLength(vehicle)-position-my_gap
                leaders.append((vehicle,gap))
            else:
                gap=position-my_length-other_pos-traci.vehicle.getMinGap(vehicle)
                followers.append((vehicle,gap))
        return leaders,followers,position

    def _lane_safe(self,target_index):
        e=self.ego;s=self.settings
        edge=traci.vehicle.getRoadID(e)
        count=traci.edge.getLaneNumber(edge)
        if target_index<0 or target_index>=count:
            return False,[], 'lane_out_of_bounds'
        target=f'{edge}_{target_index}'
        vehicle_class=traci.vehicle.getVehicleClass(e)
        allowed=traci.lane.getAllowed(target);disallowed=traci.lane.getDisallowed(target)
        if (allowed and vehicle_class not in allowed) or vehicle_class in disallowed:
            return False,[], 'lane_permission'
        if edge in self.env.spawn_edges and traci.vehicle.getLanePosition(e)<s.spawn_guard:
            return False,[], 'spawn_boundary'
        leaders,followers,position=self._neighbors(target_index)
        v=traci.vehicle.getSpeed(e);my_decel=traci.vehicle.getDecel(e)
        for leader,gap in leaders:
            if not leader:continue
            lv=traci.vehicle.getSpeed(leader)
            need=self._front_merge_gap(v,leader)
            if gap<need:
                return False,leaders,'target_leader_gap'
        for follower,gap in followers:
            if not follower:continue
            fv=traci.vehicle.getSpeed(follower)
            secure=traci.vehicle.getSecureGap(follower,fv,v,my_decel,e)
            need=max(s.lane_buffer+s.lane_time_gap*fv+max(fv-v,0)*self.dt,secure+s.lane_buffer)
            if gap<need:
                return False,leaders,'target_follower_gap'
        # A vehicle on the other side may enter the same target lane this step.
        current=traci.vehicle.getLaneIndex(e)
        far=target_index+(1 if target_index>current else -1)
        if 0<=far<count:
            toward=1 if target_index>far else -1
            flag=tc.LCA_LEFT if toward>0 else tc.LCA_RIGHT
            for vehicle in traci.lane.getLastStepVehicleIDs(f'{edge}_{far}'):
                if abs(traci.vehicle.getLanePosition(vehicle)-position)>15+v*self.dt:
                    continue
                change=traci.vehicle.getLaneChangeState(vehicle,toward)[0]
                if change&flag and not change&tc.LCA_BLOCKED:
                    return False,leaders,'competing_lane_change'
        return True,leaders,'clear'

    def _front_merge_gap(self,v,leader):
        s=self.settings
        lv=traci.vehicle.getSpeed(leader)
        secure=traci.vehicle.getSecureGap(self.ego,v,lv,traci.vehicle.getDecel(leader),leader)
        return max(s.lane_buffer+s.lane_time_gap*v+max(v-lv,0)*self.dt,secure+s.lane_buffer)

    def _lane_plan(self):
        e=self.ego;edge=traci.vehicle.getRoadID(e)
        if edge.startswith(':'):
            return dict(command=0,offset=0,goal=traci.vehicle.getLaneIndex(e),leaders=[],status='inside_junction')
        current=traci.vehicle.getLaneIndex(e)
        lane=traci.vehicle.getLaneID(e)
        best=next((b for b in traci.vehicle.getBestLanes(e) if b[0]==lane),None)
        offset=int(best[3]) if best else 0
        # An offset may recommend preparing for a later road. Do not stop at
        # this road's end when the present lane still continues along the route.
        required=bool(best is not None and not best[4])
        if not offset:
            return dict(command=0,offset=0,goal=current,leaders=[],status='route_lane_ready')
        goal=current+offset;target=current+(1 if offset>0 else -1)
        safe,leaders,status=self._lane_safe(target)
        return dict(command=(1 if offset>0 else -1) if safe else 0,
                    offset=offset,goal=goal,leaders=leaders,status=status,required=required)

    def _leader_bound(self,v,leader,gap,prepare_merge=False):
        s=self.settings
        lv=traci.vehicle.getSpeed(leader)
        decel=max(traci.vehicle.getDecel(leader),traci.vehicle.getEmergencyDecel(leader))
        margin=s.front_buffer+s.speed_buffer_time*v
        if prepare_merge:
            # Following too close for our own lane-change acceptance rule
            # creates a permanent "follow but never merge" deadlock.
            margin=max(margin,self._front_merge_gap(v,leader)+0.5)
        physical=stopping_speed(gap,s.comfortable_brake,self.dt,margin,lv,decel)
        native=traci.vehicle.getFollowSpeed(self.ego,v,max(gap,0.0),lv,decel,leader)
        return min(physical,native)

    def _signal_bound(self,v,distance,state,key=None):
        if state not in 'rRuyY':
            if key is not None:self._yellow_modes.pop(key,None)
            return math.inf
        if state in 'yY':
            saved=self._yellow_modes.get(key) if key is not None else None
            if saved is not None and self._step_time-saved[1]<=30:
                must_stop=saved[0]
            else:
                required=minimum_stopping_distance(v,self.env.max_decel,self.dt)
                must_stop=required<=max(0,distance-self.settings.signal_buffer)
                if key is not None:self._yellow_modes[key]=(must_stop,self._step_time)
            if not must_stop:
                return math.inf  # physically unable to stop before the yellow line
        elif key is not None:
            self._yellow_modes.pop(key,None)
        return stopping_speed(distance,self.settings.comfortable_brake,self.dt,self.settings.signal_buffer)

    def _exit_blocked(self,link):
        e=self.ego;lane=link[0]
        needed=traci.vehicle.getLength(e)+traci.vehicle.getMinGap(e)+2.0
        for vehicle in traci.lane.getLastStepVehicleIDs(lane):
            if vehicle==e:continue
            rear=traci.vehicle.getLanePosition(vehicle)-traci.vehicle.getLength(vehicle)
            if rear<needed and traci.vehicle.getSpeed(vehicle)<0.5:
                return True
        return False

    def predict(self,obs=None,deterministic=True):
        e=self.ego
        if e not in traci.vehicle.getIDList():
            return np.zeros(2,dtype=np.float32)
        episode=getattr(self.env,'episode',0)
        key=(episode,traci.simulation.getTime())
        if key==self._decision_key:
            return self._last_action.copy()
        if episode!=self._episode:
            self._last_acceleration=0.0;self._previous_speed=None
            self._episode=episode;self._yellow_modes.clear()
        self._step_time=key[1]
        s=self.settings;v=traci.vehicle.getSpeed(e)
        if self._previous_speed is not None and self._decision_key is not None:
            elapsed=max(self.dt,key[1]-self._decision_key[1])
            self._last_acceleration=(v-self._previous_speed)/elapsed
        lane=traci.vehicle.getLaneID(e);inside=lane.startswith(':')
        plan=self._lane_plan()
        bounds=dict(speed_limit=min(traci.vehicle.getAllowedSpeed(e),self.env.vmax))
        lead=traci.vehicle.getLeader(e,self.lookahead)
        if lead and lead[0]:
            bounds['current_leader']=self._leader_bound(v,*lead)
        prepare_merge=(plan['offset'] and plan.get('required',True)
                       and plan['status']=='target_leader_gap')
        if plan['command'] or prepare_merge:
            for index,(leader,gap) in enumerate(plan['leaders']):
                if leader:
                    prefix='merge_leader' if prepare_merge and not plan['command'] else 'target_leader'
                    bounds[f'{prefix}_{index}']=self._leader_bound(v,leader,gap,prepare_merge=prepare_merge)
        for index,tls in enumerate(traci.vehicle.getNextTLS(e)[:2]):
            cap=self._signal_bound(v,tls[2],tls[3],('tls',tls[0],int(tls[1])))
            if math.isfinite(cap):bounds[f'signal_{index}']=cap
        next_edge=self._next_edge()
        if not inside:
            room=traci.lane.getLength(lane)-traci.vehicle.getLanePosition(e)
            if plan['offset'] and plan.get('required',True):
                reserve=min(20.0,s.minimum_merge_room+3.0*max(abs(plan['offset'])-1,0))
                bounds['route_merge_room']=stopping_speed(room,s.comfortable_brake,self.dt,reserve)
            target_lane=f"{traci.vehicle.getRoadID(e)}_{traci.vehicle.getLaneIndex(e)+plan['command']}"
            target_room=traci.lane.getLength(target_lane)-traci.vehicle.getLanePosition(e)
            links=self._links(target_lane,next_edge)
            for index,link in enumerate(links):
                cap=self._signal_bound(v,min(room,target_room),link[5],('lane',target_lane,next_edge))
                if math.isfinite(cap):bounds[f'target_lane_signal_{index}']=cap
                if self._exit_blocked(link):
                    bounds[f'blocked_exit_{index}']=stopping_speed(room,s.comfortable_brake,self.dt,s.signal_buffer)
                if link[5] not in 'rRuyY' and not link[2] and room<max(25,v*self.dt+v*v/(2*s.comfortable_brake)+5):
                    bounds[f'junction_not_open_{index}']=stopping_speed(room,s.comfortable_brake,self.dt,s.signal_buffer)
            # Reuse the existing conflict geometry until benchmark evidence rejects it.
            for index,distance in enumerate(self._blocked_conflicts(self.env.junction_range,v)):
                bounds[f'conflict_{index}']=stopping_speed(distance,s.comfortable_brake,self.dt,self.yield_margin)
            if room<max(self.env.junction_range,v*self.dt+v*v/(2*s.comfortable_brake)+10):
                for index,distance in enumerate(self._yield_points(self.env.junction_range)):
                    bounds[f'yield_{index}']=stopping_speed(min(room,distance),s.comfortable_brake,self.dt,self.yield_margin)
        limit=min(bounds.values())
        desired=min(limit,v+s.acceleration*self.dt)
        acceleration=(desired-v)/self.dt
        smooth=float(np.clip(acceleration,self._last_acceleration-s.jerk_limit*self.dt,self._last_acceleration+s.jerk_limit*self.dt))
        desired=max(0.0,min(limit,v+smooth*self.dt))
        requested=(desired-v)/self.dt
        bounded=float(np.clip(requested,-self.env.max_decel,self.env.max_accel))
        action=np.asarray([bounded/(self.env.max_accel if bounded>=0 else self.env.max_decel),plan['command']],dtype=np.float32)
        self._last_acceleration=bounded;self._previous_speed=v
        self._decision_key=key;self._last_action=action.copy()
        self.last_decision=dict(target_speed=max(0,desired),reachable_speed=max(0,v+bounded*self.dt),
            limiting_reason=min(bounds,key=bounds.get),bounds={k:float(x) for k,x in bounds.items()},
            route_offset=plan['offset'],goal_lane=plan['goal'],lane_status=plan['status'],
            lane_command=plan['command'],emergency_required=limit<max(0,v-self.env.max_decel*self.dt)-1e-6)
        return action
