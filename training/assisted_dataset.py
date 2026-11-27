"""Quality filtering for assisted-teacher demonstrations."""
from collections import Counter
from pathlib import Path
import os

import numpy as np


def clean_episode(record):
    return (record.get('ended_by')=='arrival' and record.get('red_lights',0)==0
            and record.get('junction_overlaps',0)==0)


def build_dataset(directory,records,output):
    directory=Path(directory);output=Path(output)
    pieces=[];accepted=[];rejected=[]
    for record in records:
        if not clean_episode(record):
            rejected.append(record['id']);continue
        with np.load(directory/f"case_{record['id']}.npz",allow_pickle=False) as source:
            valid=source['valid_label'].astype(bool)
            if (source['collided'].any() or source['teleported'].any()
                    or source['red_light'].any()):
                raise ValueError(f"Episode {record['id']} summary and safety events disagree")
            if not valid.any():continue
            item={k:source[k][valid] for k in ('observation','next_observation','action_raw',
                  'action_requested','episode','step','terminated','truncated','arrived',
                  'collided','teleported','red_light','level')}
        obs=item['observation'];act=item['action_raw']
        if obs.shape!=(len(act),64) or act.shape!=(len(obs),2):
            raise ValueError('Unexpected observation/action shapes')
        if not np.isfinite(obs).all() or not np.isfinite(act).all() or (np.abs(act)>1).any():
            raise ValueError('Invalid observations/actions')
        if not np.isin(act[:,1],[-1,0,1]).all():raise ValueError('Non-discrete lateral labels')
        pieces.append(item);accepted.append(record)
    if not pieces:raise ValueError('No clean demonstrations')
    data={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file,**data);file.flush();os.fsync(file.fileno())
    temporary.replace(output)
    labels=data['action_raw']
    return dict(samples=len(labels),accepted_episodes=len(accepted),
        accepted_by_level=dict(Counter(r['level'] for r in accepted)),
        accepted_ids=[r['id'] for r in accepted],rejected_ids=rejected,
        lane_labels={name:int((labels[:,1]==value).sum())
                     for name,value in [('right',-1),('keep',0),('left',1)]},
        labels_different_from_requested=int((np.abs(labels-data['action_requested'])>1e-4).any(axis=1).sum()),
        observation_dim=64,label_schema='assisted64_effective_actions_v1',
        note='Only valid effective controls from clean arrivals; ambiguous road boundaries and unreachable acceleration labels excluded.')
