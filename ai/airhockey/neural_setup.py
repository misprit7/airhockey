"""Explicit simulation configuration carried with checkpoints; no hardware I/O."""
import json
from pathlib import Path
from airhockey.dynamics import _geom as geom


def workspace_bounds(profile):
    if profile=='legacy':return None
    if profile!='rail30':raise ValueError('unknown workspace profile')
    return [geom.WS_PROBE_MIN_X,geom.WS_PROBE_MAX_X,geom.WS_PROBE_MIN_Y,geom.WS_PROBE_MAX_Y]


def checkpoint_environment(path,state=None):
    meta_path=Path(path).parent/'run.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    args=state.get('args',{}) if state is not None else meta.get('args',{})
    result=dict(accel=args.get('accel',meta.get('physical_limits',{}).get('acceleration_m_s2',60)),
                defense_max_speed=args.get('defense_max_speed',12),
                workspace_bounds_mm=meta.get('workspace_bounds_mm',workspace_bounds(args.get('workspace','legacy'))))
    model=meta.get('thermal_model')
    if model:result['thermal_path']=model
    elif args.get('thermal_model'):result['thermal_path']=args['thermal_model']
    return result
