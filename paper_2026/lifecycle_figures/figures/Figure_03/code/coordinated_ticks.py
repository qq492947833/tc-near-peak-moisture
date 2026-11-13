"""Display-only fixed major ticks; original limits may expand, never shrink."""
import json
from pathlib import Path
RULES = {'a': [50, 54, 58, 62], 'b': [6, 8, 10, 12], 'c': [20, 25, 30, 35]}
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
