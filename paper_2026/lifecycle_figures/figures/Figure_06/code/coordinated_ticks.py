"""Display-only fixed major ticks; original limits may expand, never shrink."""
import json
from pathlib import Path
RULES = {'a': [75, 80, 85, 90, 95], 'b': [52, 54, 56, 58, 60], 'c': [65, 75, 85, 95, 105], 'd': [2, 2.4, 2.8, 3.2, 3.6], 'e': [0, 0.00015, 0.0003, 0.00045, 0.0006], 'f': [-0.0001, -5e-05, 0, 5e-05, 0.0001]}
AXIS = 'y'
def apply(fig,out):
    rows=[]
    for ax in fig.axes:
        key=ax.get_gid()
        if key not in RULES:continue
        old=list(getattr(ax,'get_'+AXIS+'lim')())
        ticks=RULES[key]
        getattr(ax,'set_'+AXIS+'ticks')(ticks)
        bounds=[min(old[0],min(ticks)),max(old[1],max(ticks))]
        getattr(ax,'set_'+AXIS+'lim')(*bounds)
        rows.append(dict(panel=key,axis=AXIS,old_limits=old,new_limits=bounds,major_ticks=ticks,data_preserved=True))
    (Path(out)/'tick_revision.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
