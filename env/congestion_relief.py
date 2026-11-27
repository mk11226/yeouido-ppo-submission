"""Rush-hour background-traffic relief at the configured intersections.

Monitoring and removal are environmental interventions, independent of the ego
policy. All thresholds use simulation time. Ego observations are unchanged.
"""
from collections import deque
from dataclasses import dataclass
import math


INTERSECTIONS=(
    ('영등포로/영중로','cluster_1180034702_1180034703_1180034704_1180034705'),
    ('영등포로/당산로','cluster_1180038202_1180038203_1180038204_1180038205'),
    ('여의대로/경인로','cluster_1180038706_1180038707_1180038708_1180038709_#4more'),
    ('영등포로터리','cluster_1180000601_1180000602_1180000603_1180000604'),
)


@dataclass(frozen=True)
class ReliefSettings:
    enabled: bool=True
    radius_m: float=300.
    window_s: float=120.
    sample_interval_s: float=1.
    stopped_speed_mps: float=.1
    stop_rate_threshold: float=.65
    low_speed_kmh: float=5.
    stopped_count_threshold: float=120.
    count_speed_kmh: float=8.
    intervention_interval_s: float=30.
    normal_fraction: float=.12
    normal_cap: int=20
    escalation_after_s: float=120.
    escalated_fraction: float=.25
    escalated_cap: int=40
    ego_protection_m: float=30.
    global_window_s: float=60.
    global_cap: int=120

    def __post_init__(self):
        for name in ('radius_m','window_s','sample_interval_s','intervention_interval_s',
                     'escalation_after_s','global_window_s'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite')
        for name in ('normal_cap','escalated_cap','global_cap'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('stopped_speed_mps','low_speed_kmh','stopped_count_threshold','count_speed_kmh','ego_protection_m'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError(f'Invalid {name}')
        for name in ('stop_rate_threshold','normal_fraction','escalated_fraction'):
            if not math.isfinite(getattr(self,name)) or not 0<=getattr(self,name)<=1:raise ValueError(f'Invalid {name}')
        if self.sample_interval_s>self.window_s:raise ValueError('Sample interval exceeds averaging window')


@dataclass(frozen=True)
class Vehicle:
    x: float
    y: float
    speed: float
    waiting: float=0.


class RollingTrafficWindow:
    def __init__(self,seconds):
        self.seconds=seconds;self.samples=deque()

    def append(self,start,end,count,stopped,speed_sum):
        if end<=start:return
        self.samples.append((start,end,count,stopped,speed_sum))
        cutoff=end-self.seconds
        while self.samples and self.samples[0][1]<=cutoff+1e-9:self.samples.popleft()

    def metrics(self,now):
        duration=rate=stopped_total=speed_total=nonempty_duration=0.
        cutoff=now-self.seconds
        for start,end,count,stopped,speed_sum in self.samples:
            dt=max(0.,min(end,now)-max(start,cutoff))
            duration+=dt
            stopped_total+=stopped*dt
            if count:
                rate+=stopped/count*dt
                speed_total+=speed_sum/count*dt
                nonempty_duration+=dt
        return dict(window_seconds=duration,window_ready=duration>=self.seconds-1e-7,
            stop_rate=rate/duration if duration else 0.,
            stopped_mean=stopped_total/duration if duration else 0.,
            speed_mean_kmh=speed_total/nonempty_duration*3.6 if nonempty_duration else None)


def congested(metrics,settings):
    speed=metrics['speed_mean_kmh']
    return bool(metrics['window_ready'] and speed is not None and (
        (metrics['stop_rate']+1e-12>=settings.stop_rate_threshold and speed<=settings.low_speed_kmh+1e-9)
        or (metrics['stopped_mean']+1e-9>=settings.stopped_count_threshold and speed<=settings.count_speed_kmh+1e-9)))


class CongestionRelief:
    """Pure coordinator: one rolling global removal budget for all centers.

    remove(id) must perform any final live-state protection check and return
    True only after a successful removal. Only successful removals consume quota.
    """
    def __init__(self,centers,settings=None):
        self.settings=settings or ReliefSettings()
        self.centers=list(centers)
        self.reset(0.,False)

    def reset(self,now,active=True):
        self.active=bool(active and self.settings.enabled)
        self.last_sample=now
        self.windows={name:RollingTrafficWindow(self.settings.window_s) for name,x,y in self.centers}
        self.first_intervention={name:None for name,x,y in self.centers}
        self.next_intervention={name:now for name,x,y in self.centers}
        self.budget=deque();self.removed_total=0;self.removed_by_center={name:0 for name,x,y in self.centers}
        self.last_metrics={};self.last_events=[];self.rotation=0

    def due(self,now):
        return self.active and now-self.last_sample>=self.settings.sample_interval_s-1e-8

    def update(self,now,vehicles,ego_id,remove):
        self.last_events=[]
        if not self.due(now):return []
        s=self.settings;start=self.last_sample;self.last_sample=now
        while self.budget and self.budget[0]<=now-s.global_window_s+1e-9:self.budget.popleft()
        ego=vehicles.get(ego_id)
        plans={};locations={name:(x,y) for name,x,y in self.centers}
        for name,x,y in self.centers:
            nearby={identifier:v for identifier,v in vehicles.items() if identifier!=ego_id
                    and (v.x-x)**2+(v.y-y)**2<=s.radius_m**2+1e-8}
            stopped={identifier:v for identifier,v in nearby.items() if v.speed<s.stopped_speed_mps}
            self.windows[name].append(start,now,len(nearby),len(stopped),sum(v.speed for v in nearby.values()))
            metrics=self.windows[name].metrics(now)
            severe=congested(metrics,s)
            if not severe:self.first_intervention[name]=None
            first=self.first_intervention[name]
            escalated=first is not None and now-first>=s.escalation_after_s-1e-8
            metrics.update(congested=severe,stopped_now=len(stopped),vehicle_count_now=len(nearby),
                stage='escalated' if escalated else 'normal',first_intervention_time=first)
            self.last_metrics[name]=metrics
            if not severe or ego is None or now<self.next_intervention[name]-1e-8:continue
            self.next_intervention[name]=now+s.intervention_interval_s
            fraction=s.escalated_fraction if escalated else s.normal_fraction
            cap=s.escalated_cap if escalated else s.normal_cap
            quota=min(cap,int(math.floor(len(stopped)*fraction+1e-9)))
            # Rank deterministically by continuous waiting time, then distance.
            eligible=[identifier for identifier,v in stopped.items()
                      if (v.x-ego.x)**2+(v.y-ego.y)**2>s.ego_protection_m**2+1e-8]
            eligible.sort(key=lambda identifier:(-stopped[identifier].waiting,
                (stopped[identifier].x-x)**2+(stopped[identifier].y-y)**2,identifier))
            metrics.update(eligible_now=len(eligible),requested_quota=quota)
            plans[name]=[deque(eligible),quota]
        # Interleave centers so a nearly exhausted global budget does not always
        # favor the first intersection. Overlapping circles cannot remove twice.
        names=[name for name,x,y in self.centers]
        names=names[self.rotation:]+names[:self.rotation]
        if plans:self.rotation=(self.rotation+1)%len(names)
        removed=set()
        while len(self.budget)<s.global_cap:
            attempted=False
            for name in names:
                if len(self.budget)>=s.global_cap:break
                if name not in plans:continue
                candidates,quota=plans[name]
                if quota<=0:continue
                while candidates and candidates[0] in removed:candidates.popleft()
                if not candidates:continue
                attempted=True;identifier=candidates.popleft()
                if not remove(identifier):continue
                removed.add(identifier);plans[name][1]-=1;self.budget.append(now)
                if self.first_intervention[name] is None:
                    self.first_intervention[name]=now
                    self.last_metrics[name]['first_intervention_time']=now
                self.removed_total+=1;self.removed_by_center[name]+=1
                v=vehicles[identifier];x,y=locations[name]
                self.last_events.append(dict(time=now,intersection=name,vehicle_id=identifier,
                    stage=self.last_metrics[name]['stage'],speed_mps=v.speed,
                    distance_to_center_m=math.hypot(v.x-x,v.y-y),
                    distance_to_ego_m=math.hypot(v.x-ego.x,v.y-ego.y),
                    window_metrics=dict(self.last_metrics[name])))
            if not attempted:break
        return self.last_events

    def info(self):
        return dict(enabled=self.active,removed_total=self.removed_total,
            removed_by_intersection=dict(self.removed_by_center),removed_last_60s=len(self.budget),
            removed_this_step=len(self.last_events),regions={name:dict(value) for name,value in self.last_metrics.items()})


class SumoCongestionRelief:
    """Efficient junction-context adapter; no all-network per-vehicle scan."""
    def __init__(self,net,config):
        self.settings=ReliefSettings(**config)
        self.junctions=[];centers=[]
        for name,node_id in INTERSECTIONS:
            node=net.getNode(node_id);x,y=node.getCoord()
            self.junctions.append(node_id);centers.append((name,x,y))
        self.controller=CongestionRelief(centers,self.settings)

    def reset(self,traci,tc,level):
        self.controller.reset(traci.simulation.getTime(),level=='rush')
        if not self.controller.active:return
        for node_id in self.junctions:
            traci.junction.subscribeContext(node_id,tc.CMD_GET_VEHICLE_VARIABLE,self.settings.radius_m,
                [tc.VAR_POSITION,tc.VAR_SPEED,tc.VAR_WAITING_TIME])

    def step(self,traci,tc,ego_id):
        controller=self.controller;controller.last_events=[]
        now=traci.simulation.getTime()
        if not controller.due(now):return []
        vehicles={}
        for node_id in self.junctions:
            for identifier,data in (traci.junction.getContextSubscriptionResults(node_id) or {}).items():
                x,y=data[tc.VAR_POSITION]
                vehicles[identifier]=Vehicle(x,y,data[tc.VAR_SPEED],data.get(tc.VAR_WAITING_TIME,0.))
        # Ego can be outside all monitoring circles; protection still applies.
        if ego_id not in traci.vehicle.getIDList():return []
        ex,ey=traci.vehicle.getPosition(ego_id)
        vehicles[ego_id]=Vehicle(ex,ey,traci.vehicle.getSpeed(ego_id))
        def remove(identifier):
            if identifier==ego_id:return False
            try:
                x,y=traci.vehicle.getPosition(identifier)
                # Re-check current speed and the full 30 m ego exclusion circle.
                if traci.vehicle.getSpeed(identifier)>=self.settings.stopped_speed_mps:return False
                px,py=traci.vehicle.getPosition(ego_id)
                if (x-px)**2+(y-py)**2<=self.settings.ego_protection_m**2+1e-8:return False
                traci.vehicle.remove(identifier)
                return True
            except traci.TraCIException:return False
        return controller.update(now,vehicles,ego_id,remove)
