"""20261028 display-only stage, numeric colorbar and category revision.
Each call runs after the unchanged pre-style scientific object gate.
"""
from pathlib import Path
from decimal import Decimal,ROUND_CEILING
import re,json,copy,hashlib
import numpy as np
from matplotlib.text import Text
from matplotlib.colors import TwoSlopeNorm,Normalize,LogNorm
from matplotlib.ticker import FixedLocator,FuncFormatter
from display_format import format_number

def outward(x):
    x=float(x)
    if x==0:return 1.
    step=Decimal('0.05')*(Decimal(10)**int(np.floor(np.log10(abs(x)))))
    return float((Decimal(str(abs(x)))/step).to_integral_value(rounding=ROUND_CEILING)*step)

def rewrite(fig):
    changes=[]
    for t in fig.findobj(Text):
        old=t.get_text();new=re.sub(r'\bResolution\b','Decay',old);new=re.sub(r'\bRes\b','Dec',new)
        if old!=new:t.set_text(new);changes.append({'old':old,'new':new})
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            fm=axis.get_major_formatter()
            axis.set_major_formatter(FuncFormatter(lambda x,pos,fm=fm:re.sub(r'\bRes\b','Dec',re.sub(r'\bResolution\b','Decay',fm(x,pos)))))
    return changes

def science(fig):
    # All original data axes; auxiliary colorbar geometry and display norms excluded.
    result=[]
    from supp_style import data_state
    states=data_state(fig)
    for ax,s in zip(fig.axes,states):
        if hasattr(ax,'_colorbar') or ax.get_label().startswith('<colorbar'):continue
        for im in s['images']:
            for k in ['norm','vmin','vmax']:im.pop(k,None)
        s['lines']=[d for d,l in zip(s['lines'],ax.lines) if l.get_gid()!='display_group_bracket']
        result.append(s)
    return result

def apply_decay_display(fig,fid,out):
    out=Path(out);before=science(fig);changes=rewrite(fig);axes={a.get_gid():a for a in fig.axes if a.get_gid()};records=[]
    if fid=='Figure_S06':
        a=axes['a'];p=a.get_position();im=a.images[0];v=np.asarray(im.get_array());lo=int(np.floor(v.min()/20)*20);hi=int(np.ceil(v.max()/20)*20)
        im.set_norm(Normalize(lo,hi));ca=fig.add_axes([p.x0+.04*p.width,.537,.92*p.width,.014]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=np.linspace(lo,hi,5));cb.set_label('Valid tropical cyclones (count)',fontsize=7.5,labelpad=2)
        cb.ax.tick_params(labelsize=7,length=2,pad=2)
        b=axes['b'];fixed_xlim=b.get_xlim()
        for t in b.texts:
            if t.get_text() in ['0.85','0.95']:t.set_position((1 if t.get_text()=='0.85' else 4,-.225))
        for x0,x1 in [(-.45,2.45),(2.55,5.45)]:
            line,=b.plot([x0,x0,x1,x1],[-.13,-.175,-.175,-.13],transform=b.get_xaxis_transform(),color='#233A46',lw=.7,clip_on=False);line.set_gid('display_group_bracket')
        b.set_xlim(fixed_xlim)
        ims=[axes[k].images[0] for k in ['c','d']];lim=outward(max(np.max(np.abs(np.asarray(x.get_array()))) for x in ims))
        norm=TwoSlopeNorm(0,-lim,lim)
        for x in ims:x.set_norm(norm)
        cb=ims[-1].colorbar;cb.set_ticks(np.linspace(-lim,lim,5));cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:format_number(Decimal(str(x)).scaleb(3))))
    if fid=='Figure_S07':
        for p in axes['c'].patches:p.set_facecolor('#7E894E')
        a=axes['d'];im=a.images[0];lim=outward(np.nanmax(abs(np.asarray(im.get_array()))));im.set_norm(TwoSlopeNorm(0,-lim,lim));a.set_xlabel('');a.set_title('Trends; † CI includes 0',loc='left')
        p=a.get_position();ca=fig.add_axes([p.x0+.07*p.width,.371,.86*p.width,.010]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=np.linspace(-lim,lim,5));cb.set_label('Trend (h per decade)',fontsize=7.2,labelpad=2);cb.ax.xaxis.set_label_position('top');cb.ax.tick_params(labelsize=7,length=2,pad=2)
        for cax in fig.axes:
            if hasattr(cax,'_colorbar') and isinstance(cax._colorbar.norm,LogNorm):cax._colorbar.set_ticks([1,10,100,1000,2340]);cax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:format_number(x,decimals=0)))
    if fid=='Figure_S07':
        # Add 6 mm of genuine inter-row clearance while preserving every data panel's physical size.
        oldh=fig.get_figheight();extra=6/25.4;positions=[(a,a.get_position().frozen())for a in fig.axes]
        fig.set_size_inches(fig.get_figwidth(),oldh+extra)
        for a,p in positions:
            upper=a.get_gid() in ['a','b','c','d'] or hasattr(a,'_colorbar') and not isinstance(a._colorbar.norm,LogNorm)
            a.set_position([p.x0,(p.y0*oldh+(extra if upper else 0))/(oldh+extra),p.width,p.height*oldh/(oldh+extra)])
    if fid=='Figure_S09':
        import pandas as pd
        pkg=Path(__file__).resolve().parents[1];df=pd.read_csv(next((pkg/'inputs').glob('*fig6_regions.csv')),float_precision='round_trip')
        from matplotlib.patches import Wedge
        cbs=[x._colorbar for x in fig.axes if hasattr(x,'_colorbar')]
        for key,field,cb in zip(['a','b'],['q700_asymmetric','cyclonic_tangential_wind700_asymmetric'],cbs):
            z=df[df.field.eq(field)&df.estimand.eq('pre_mean')&df.sector.isin(['downshear','left_of_shear','upshear','right_of_shear'])]
            vals=[float(z[z.region.eq(r)&z.sector.eq(s)].estimate.iloc[0]) for r in ['100_300km','300_800km'] for s in ['downshear','left_of_shear','upshear','right_of_shear']]
            lim=outward(max(abs(np.array(vals))));norm=TwoSlopeNorm(0,-lim,lim);cb.mappable.set_norm(norm)
            for patch,val in zip([p for p in axes[key].patches if isinstance(p,Wedge)],vals):patch.set_facecolor(cb.cmap(norm(val)))
            cb.mappable._frozen_values=np.array(vals);cb.set_ticks(np.linspace(-lim,lim,5));p=cb.ax.get_position();cb.ax.set_position([p.x0,p.y0+.028,p.width,p.height])
    # Record every numeric colour scale and its real endpoints, after formatting.
    for ca in fig.axes:
        if not hasattr(ca,'_colorbar'):continue
        cb=ca._colorbar;norm=cb.norm
        related=[(a,m) for a in fig.axes if not hasattr(a,'_colorbar') for m in [*a.images,*a.collections] if getattr(m,'norm',None) is norm]
        arrs=[np.ma.asarray(m.get_array()).compressed() for a,m in related if m.get_array() is not None]
        if hasattr(cb.mappable,'_frozen_values'):arrs.append(cb.mappable._frozen_values)
        arr=np.concatenate(arrs) if arrs else np.array([]);arr=arr[np.isfinite(arr)]
        if isinstance(norm,LogNorm):ticks=np.array([1,10,100,1000,norm.vmax]);ticks=np.unique(ticks[(ticks>=norm.vmin)&(ticks<=norm.vmax)])
        else:ticks=np.linspace(norm.vmin,norm.vmax,5)
        cb.set_ticks(ticks)
        e=-3 if fid=='Figure_S06' and norm.vmin<0 else 0
        formatter=FuncFormatter(lambda x,p,e=e:format_number(Decimal(str(x)).scaleb(-e)))
        (ca.xaxis if cb.orientation=='horizontal' else ca.yaxis).set_major_formatter(formatter)
        assert not arr.size or arr.min()>=norm.vmin-1e-12 and arr.max()<=norm.vmax+1e-12
        records.append({'panels':list(dict.fromkeys(a.get_gid() for a,m in related)) or (['a'] if len(records)==0 and fid=='Figure_S09' else ['b'] if fid=='Figure_S09' else []),'semantic':ca.get_xlabel() or ca.get_ylabel(),'data_range':[float(arr.min()),float(arr.max())] if arr.size else None,'normalization':type(norm).__name__,'norm_range':[norm.vmin,norm.vmax],'ticks':ticks.tolist(),'endpoints_present':True,'cmap':cb.cmap.name,'clipping':False,'scientific_values_rescaled':False,'display_exponent':e})
    fig.canvas.draw();after=science(fig)
    if before!=after:
        (out/'post_style_science_difference.json').write_text(json.dumps({'before':before,'after':after},ensure_ascii=False),encoding='utf8')
        raise AssertionError((fid,'scientific data/CI/geometry changed'))
    left=[t.get_text()for t in fig.findobj(Text) if re.search(r'\b(?:Resolution|Res)\b',t.get_text())];assert not left,left
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    categorical={
      'Figure_S02':'Stage categories D/E/F/Dec; basin categories NA/EP/WP/NI/SI/SP, discrete legends.',
      'Figure_S03':'Window categories and paired-contrast classes; discrete legends.',
      'Figure_S04':'Stage categories, directly labelled D/E/F/Dec and respective contrasts.',
      'Figure_S05':'Representation schemes, held-out models and response diagnostics; direct labels and discrete legend.',
      'Figure_S06':'Later L / Bridge B classes in b; numerical counts a; signed differences c/d; dagger denotes CI includes zero.',
      'Figure_S07':'Return classes a, episode/return categories b, uniform bridge-duration bars c; numeric trends d with dagger CI annotation; numeric log-count histograms e/f.',
      'Figure_S08':'Transition classes and observed/background line styles; discrete legend.',
      'Figure_S09':'Numeric moisture/circulation differences, separate units and scales.'}
    report={'figure':fid,'status':'PASS_DISPLAY_SCIENTIFIC_GUARD','scientific_data_before_sha256':digest(before),'scientific_data_after_sha256':digest(after),'all_scientific_values_CI_geometry_equal':True,'allowed_changes':'display strings, fill colors, norms and colorbar axes/ticks, grouping brackets, positions only','stage_label_changes':changes,'color_semantics':categorical[fid],'numeric_colorbars':records,'visual_status':'PENDING_ACTUAL_REVIEW'}
    (out/'decay_colorbar_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
