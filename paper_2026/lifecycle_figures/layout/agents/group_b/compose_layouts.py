"""Lossless assembly of approved complete panels; no redraw, resampling or text edits."""
from pathlib import Path
import sys,itertools
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts'))
from common import *
from PIL import Image
import numpy as np
WORK=R/'agents/group_b'
OUT=WORK/'output';OUT.mkdir(exist_ok=True)
PREVIEW=WORK/'previews';PREVIEW.mkdir(exist_ok=True)

def assemble(stem,panels,size,notes,gaps):
 source=R/'backup'/f'{stem}.png';im=Image.open(source).convert('RGBA');a=np.array(im)
 assert np.all(a[:,:,3]==255)
 ink=np.any(a[:,:,:3]!=255,axis=2)
 covered=np.zeros(ink.shape,dtype=bool)
 result=Image.new('RGBA',size,(255,255,255,255));rects=[]
 for p in panels:
  box=tuple(p['source_bbox']);x,y=p['destination_xy'];w,h=box[2]-box[0],box[3]-box[1]
  target=(x,y,x+w,y+h);assert 0<=x and 0<=y and target[2]<=size[0]and target[3]<=size[1]
  assert not covered[box[1]:box[3],box[0]:box[2]].any()
  covered[box[1]:box[3],box[0]:box[2]]=True
  for old in rects:
   assert min(target[2],old[2])<=max(target[0],old[0])or min(target[3],old[3])<=max(target[1],old[1]),(target,old)
  patch=im.crop(box);result.paste(patch,(x,y));rects.append(target)
  p.update(destination_bbox=list(target),pixel_scale=1.0,source_crop_rgba_sha256=hashlib.sha256(patch.tobytes()).hexdigest())
 assert not (ink&~covered).any(), 'Non-white source pixels omitted'
 for p in panels:
  assert im.crop(tuple(p['source_bbox'])).tobytes()==result.crop(tuple(p['destination_bbox'])).tobytes()
 ar=np.array(result);ink_after=np.any(ar[:,:,:3]!=255,axis=2)
 assert int(ink.sum())==int(ink_after.sum())
 # Retain original physical pixel sampling. This is a smaller canvas, never upsampling.
 result.save(OUT/f'{stem}.png',dpi=(900,900))
 result.save(OUT/f'{stem}.tiff',dpi=(900,900),compression='tiff_lzw')
 with Image.open(OUT/f'{stem}.png')as p,Image.open(OUT/f'{stem}.tiff')as t:assert p.convert('RGBA').tobytes()==t.convert('RGBA').tobytes()==result.tobytes()
 preview=result.convert('RGB');preview.thumbnail((1500,1500));preview.save(PREVIEW/f'{stem}_after.png')
 rec={'figure':stem,'source_path':str(source),'source_sha256':sha(source),'source_size':list(im.size),'output_size':list(size),'panels':panels,'notes':notes,'gaps_pixels':gaps,'resampling':False,'pixel_scale':1,'source_nonwhite_pixels':int(ink.sum()),'output_nonwhite_pixels':int(ink_after.sum()),'all_nonwhite_source_pixels_covered':True,'every_complete_crop_exactly_preserved':True,'destination_crops_disjoint':True,'png_tiff_pixel_identical':True,'output_png_sha256':sha(OUT/f'{stem}.png'),'output_tiff_sha256':sha(OUT/f'{stem}.tiff')}
 dump(WORK/f'{stem}_layout.json',rec)
 return rec

records=[]
# Three rows x two columns. Common column boundaries preserve all axis alignment.
panels=[]
row_bands=[(207,2136),(2657,4586),(5105,7036)]
yb=80
for row,(top,bottom)in enumerate(row_bands):
 for col,(left,right)in enumerate([(93,3003),(3546,6192)]):
  panels.append({'id':'abcdef'[row*2+col],'source_bbox':[left,top,right,bottom],'destination_xy':[80 if col==0 else 3190,yb]})
 yb+=bottom-top+200
records.append(assemble('Figure_S05',panels,(5916,6349),'Six full panels retained at original pixel size. Two column boundaries and three row origins remain common, including long labels in panel e. Cropped only pure outer white and empty gutters.',{'horizontal_before_min':543,'horizontal_after':200,'vertical_before':[521,519],'vertical_after':[200,200],'outer_margin_after':80}))
# Two-panel upper row and a full-span lower panel. The full-width c panel is untouched.
panels=[
 {'id':'a_with_period_legend','source_bbox':[622,167,3162,2873],'destination_xy':[187,80]},
 {'id':'b','source_bbox':[3681,167,6254,2873],'destination_xy':[2927,80]},
 {'id':'c_full_width','source_bbox':[728,3234,6255,5130],'destination_xy':[80,2946]},
]
records.append(assemble('Figure_S07',panels,(5687,4922),'Upper a/b complete-panel group centered over unchanged full-width c. Shared period legend travels with a. No part of c axis is removed or stretched. Upper group and lower full-span panel are not an equal-column grid, so left plot-area edges intentionally differ after centering.',{'horizontal_upper_before':519,'horizontal_upper_after':200,'vertical_groups_before':361,'vertical_groups_after':160,'outer_margin_after':80,'upper_group_left_right_inset_relative_to_c':[107,107]}))
dump(WORK/'LAYOUTS.json',records)
print(json.dumps([{k:r[k]for k in['figure','source_size','output_size','source_nonwhite_pixels','output_nonwhite_pixels','gaps_pixels']}for r in records],ensure_ascii=False))
