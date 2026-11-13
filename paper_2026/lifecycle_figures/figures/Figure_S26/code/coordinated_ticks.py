"""Display-only fixed major ticks; original limits may expand, never shrink."""
import json
from pathlib import Path
RULES = {'a': [-2.5, 0, 2.5, 5, 7.5], 'b': [0, 0.2, 0.4, 0.6, 0.8], 'c': [0, 5e-05, 0.0001, 0.00015, 0.0002], 'd': [-4e-05, 0, 4e-05, 8e-05, 0.00012], 'e': [-2.4e-05, -1.8e-05, -1.2e-05, -6e-06, 0], 'f': [-0.0004, -0.0003, -0.0002, -0.0001, 0]}
AXIS = 'x'
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
