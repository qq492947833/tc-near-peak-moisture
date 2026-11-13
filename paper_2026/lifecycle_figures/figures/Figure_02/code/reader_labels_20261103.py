"""Clarify adjacent-stage cyclone-pair estimators without changing input categories."""
from pathlib import Path
import json
import numpy as np

def apply(fig,out):
    oldw,oldh=fig.get_size_inches()*25.4
    before=[np.asarray(a.get_position().bounds)*[oldw,oldh,oldw,oldh] for a in fig.axes]
    legend_positions=[(l,l.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)for l in fig.legends]
    changes=[]
    mapping={'Raw, full support':'All valid stage pairs\nUnadjusted',
             'Raw, common support':'Geographic/seasonal overlap\nUnadjusted',
             'Adjusted, common support':'Same overlap subset\nAdjusted for location/season'}
    for legend in fig.legends:
        for t in legend.get_texts():
            if t.get_text()in mapping:
                old=t.get_text();t.set_text(mapping[old]);changes.append(dict(before=old,after=t.get_text()))
    assert len(changes)==3
    # The clearer two-line legend gets 4 mm below the unchanged physical panels.
    newh=oldh+4
    fig.set_size_inches(oldw/25.4,newh/25.4,forward=False)
    for a,(x,y,w,h)in zip(fig.axes,before):
        a.set_axes_locator(None);a.set_position([x/oldw,(y+4)/newh,w/oldw,h/newh])
    for l,(x,y,w,h)in legend_positions:
        dy=0 if any(t.get_text()in mapping.values()for t in l.get_texts()) else 4
        l.set_bbox_to_anchor((x,(y*oldh+dy)/newh,w,h*oldh/newh),transform=fig.transFigure)
    fig.canvas.draw()
    after=[np.asarray(a.get_position().bounds)*[oldw,newh,oldw,newh]for a in fig.axes]
    for b,a in zip(before,after):assert np.allclose(a,b+[0,4,0,0],atol=1e-12,rtol=0)
    report=dict(status='PASS_LABEL_ONLY',changes=changes,axes_physical_size_and_top_position_exact=True,
        original_size_mm=[oldw,oldh],new_size_mm=[oldw,newh],extra_bottom_canvas_mm=4,
        display_input_unchanged=True,unit='Same-SID adjacent-stage pairs; not departure-event pairs',
        original_support='All valid adjacent-stage pairs, n_pairs from raw_adjacent_contrasts.csv',
        overlap_support='Same unadjusted/adjusted geographic-seasonal overlap subset in adjusted_adjacent_summary.csv; n_unique equals n_pairs',
        adjusted_statistic='Saved geography-adjusted estimator; no new weighting or fitting performed',
        units='Each native contrast and CI divided by its fixed whole-lifecycle reference SD')
    (Path(out)/'SEMANTIC_REVISION_QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
