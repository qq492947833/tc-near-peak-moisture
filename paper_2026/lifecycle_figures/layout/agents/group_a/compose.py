"""Reassemble accepted full-panel pixels; no plotting or resampling.

The only omitted source pixels are opaque white. All image content is retained
within disjoint full-panel rectangles at its original sampling density.
"""
from pathlib import Path
import runpy,sys,hashlib,json,copy
R=Path(__file__).resolve().parents[2]
from PIL import Image
import numpy as np
OUT=R/'agents/group_a/output';OUT.mkdir(exist_ok=True)
PRE=R/'agents/group_a/previews';PRE.mkdir(exist_ok=True)
def shafile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def shabytes(a):return hashlib.sha256(a.tobytes()).hexdigest()
specs={
 'Figure_07':{'column_source_bounds':[(149,3020),(3402,6255)],'row_source_bounds':[(118,1930),(2218,4030),(4318,6130)],'column_gap':150,'row_gap':105,'margin':70,'labels':[['a','b'],['c','d'],['e','f']]},
 'Figure_S06':{'column_source_bounds':[(407,2956),(3609,6192)],'row_source_bounds':[(186,2619),(3146,5579)],'column_gap':210,'row_gap':180,'margin':70,'labels':[['a','b'],['c','d']]},
}
report=[]
for stem,spec in specs.items():
 src=R/'backup'/f'{stem}.png';im=Image.open(src).convert('RGBA');a=np.array(im);bg=np.array([255,255,255,255],dtype=np.uint8)
 assert np.array_equal(a[0,0],bg)
 xs=spec['column_source_bounds'];ys=spec['row_source_bounds'];m=spec['margin'];gx=spec['column_gap'];gy=spec['row_gap']
 widths=[b-a for a,b in xs];heights=[b-a for a,b in ys]
 W=sum(widths)+gx*(len(xs)-1)+2*m;H=sum(heights)+gy*(len(ys)-1)+2*m
 canvas=Image.new('RGBA',(W,H),tuple(map(int,bg)))
 coverage=np.zeros(a.shape[:2],bool);placements=[];dst_coverage=np.zeros((H,W),bool)
 for ri,(y0,y1)in enumerate(ys):
  for ci,(x0,x1)in enumerate(xs):
   srcbox=[x0,y0,x1,y1];x=m+sum(widths[:ci])+ci*gx;y=m+sum(heights[:ri])+ri*gy
   block=im.crop(srcbox);canvas.paste(block,(x,y))
   assert not coverage[y0:y1,x0:x1].any();coverage[y0:y1,x0:x1]=True
   assert not dst_coverage[y:y+block.height,x:x+block.width].any();dst_coverage[y:y+block.height,x:x+block.width]=True
   placements.append({'panel':spec['labels'][ri][ci],'source_box':srcbox,'destination_box':[x,y,x+block.width,y+block.height],'pixel_scale':1.0,'source_pixel_sha256':shabytes(np.array(block))})
 omitted=a[~coverage];assert np.all(omitted==bg),'Any nonwhite source omitted'
 dest=np.array(canvas)
 for p in placements:
  x0,y0,x1,y1=p['destination_box'];assert shabytes(dest[y0:y1,x0:x1])==p['source_pixel_sha256']
 assert int(np.any(a!=bg,axis=2).sum())==int(np.any(dest!=bg,axis=2).sum())
 dpi=Image.open(src).info['dpi'];png=OUT/f'{stem}.png';tif=OUT/f'{stem}.tiff'
 canvas.save(png,dpi=dpi);canvas.save(tif,dpi=dpi,compression='tiff_lzw')
 assert np.array_equal(np.asarray(Image.open(png)),np.asarray(Image.open(tif)))
 prev=canvas.convert('RGB');prev.thumbnail((1500,1500));prev.save(PRE/f'{stem}_after.png')
 originalcolgap=xs[1][0]-xs[0][1];originalrowgaps=[ys[i+1][0]-ys[i][1]for i in range(len(ys)-1)]
 row={'stem':stem,'source_sha256':shafile(src),'source_size':list(im.size),'output_size':list(canvas.size),'dpi':list(dpi),'column_gap_before_px':originalcolgap,'column_gap_after_px':gx,'row_gaps_before_px':originalrowgaps,'row_gaps_after_px':[gy]*len(originalrowgaps),'outer_margin_after_px':m,'output_png_sha256':shafile(png),'output_tiff_sha256':shafile(tif),'placements':placements,'source_nonwhite_pixels':int(np.any(a!=bg,axis=2).sum()),'output_nonwhite_pixels':int(np.any(dest!=bg,axis=2).sum()),'omitted_source_pixels_all_exact_opaque_white':True,'pixel_identity_each_complete_panel':True,'output_rectangles_disjoint':True,'PNG_TIFF_pixel_identical':True,'no_resampling':True,'width_reduction_percent':100*(1-W/im.width),'height_reduction_percent':100*(1-H/im.height)}
 report.append(row)
 print(stem,im.size,canvas.size,'horizontal_gap',originalcolgap,'->',gx,'vertical_gap',originalrowgaps,'->',gy)
source=R/'provenance/Figure_LATITUDE.png'
source_match=np.array_equal(np.asarray(Image.open(source).convert('RGBA')),np.asarray(Image.open(R/'backup/Figure_S06.png').convert('RGBA')))
report[1]['existing_complete_vector_source']=str(source.with_suffix('.svg'))
report[1]['existing_vector_raster_companion_matches_current_pixels']=source_match
report[1]['source_choice']='Accepted final raster panels are moved at1:1 to guarantee byte-for-byte scientific pixels. Existing complete SVG provenance retained; no final standalone panel assets were found in that output folder.'
(R/'agents/group_a/layout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
