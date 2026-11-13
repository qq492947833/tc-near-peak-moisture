from pathlib import Path
import runpy,json,hashlib
R=Path(__file__).resolve().parents[2]
W=R/'agents/group_c';O=W/'output';O.mkdir(exist_ok=True)
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,blocks,size,gaps):
 src=Image.open(R/'backup'/f'{name}.png').convert('RGBA');a=np.asarray(src)
 assert (a[:,:,3]==255).all()
 ink=(a[:,:,:3]!=255).any(2);covered=np.zeros(ink.shape,dtype=bool)
 result=Image.new('RGBA',size,(255,255,255,255));destcov=np.zeros((size[1],size[0]),dtype=bool)
 items=[]
 for label,box,xy in blocks:
  x0,y0,x1,y1=box;dx,dy=xy;ww=x1-x0;hh=y1-y0
  assert not covered[y0:y1,x0:x1].any(),label
  assert not destcov[dy:dy+hh,dx:dx+ww].any(),label
  assert dx>=0 and dy>=0 and dx+ww<=size[0] and dy+hh<=size[1]
  covered[y0:y1,x0:x1]=True;destcov[dy:dy+hh,dx:dx+ww]=True
  crop=src.crop(box);result.paste(crop,xy)
  assert result.crop((dx,dy,dx+ww,dy+hh)).tobytes()==crop.tobytes()
  items.append({'label':label,'source_box':list(box),'destination_xy':list(xy),'destination_box':[dx,dy,dx+ww,dy+hh],'pixels_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),'source_nonwhite_pixels':int(ink[y0:y1,x0:x1].sum())})
 assert not ink[~covered].any(),int(ink[~covered].sum())
 new=np.asarray(result);newink=(new[:,:,:3]!=255).any(2)
 assert ink.sum()==newink.sum()
 result.save(O/f'{name}.png',dpi=(900,900))
 result.save(O/f'{name}.tiff',dpi=(900,900),compression='tiff_lzw')
 png=Image.open(O/f'{name}.png').convert('RGBA');tif=Image.open(O/f'{name}.tiff').convert('RGBA')
 assert png.tobytes()==tif.tobytes()==result.tobytes()
 preview=result.convert('RGB');preview.thumbnail((1500,1500));preview.save(O/f'{name}_preview.png')
 # Separate high-resolution geometry manifest uses only translation; all pixel blocks remain unchanged.
 info={'figure':name,'source':str(R/'backup'/f'{name}.png'),'source_sha256':sha(R/'backup'/f'{name}.png'),'source_size':list(src.size),'output_size':list(size),'native_dpi':900,'scale_factor':1.0,'operation':'lossless complete-panel crop and translation; no interpolation or redrawing','blocks':items,'gap_changes_pixels':gaps,'all_nonwhite_source_pixels_preserved':True,'all_RGBA_source_blocks_pixel_identical':True,'nonwhite_count':int(ink.sum()),'source_crop_overlap':False,'destination_overlap':False,'PNG_TIFF_RGBA_equal':True,'outputs':{'png':sha(O/f'{name}.png'),'tiff':sha(O/f'{name}.tiff')},'area_reduction_percent':100*(1-size[0]*size[1]/(src.width*src.height))}
 (W/f'{name}_layout.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 return info

blocks=[]
xs=[(210,3022),(3694,6211)];ys=[(207,2136),(2657,4586),(5105,7036)];dstx=[50,3022];dsty=[50,2159,4268]
for r,(y0,y1)in enumerate(ys):
 for c,(x0,x1)in enumerate(xs):blocks.append((chr(ord('a')+2*r+c),(x0,y0,x1,y1),(dstx[c],dsty[r])))
s08=save('Figure_S08',blocks,(5589,6249),{'horizontal_minimum':[672,160],'row1_row2':[521,180],'row2_row3':[519,180],'outer_left':[210,50],'outer_top':[207,50],'outer_right':[166,50],'outer_bottom':[227,50]})
blocks=[('shared_legend',(2362,114,4385,212),(1818,50))]
xs=[(208,2060),(2478,4110),(4448,6164)];ys=[(535,2588),(2854,4907)]
for r,(y0,y1)in enumerate(ys):
 for c,(x0,x1)in enumerate(xs):blocks.append((chr(ord('a')+3*r+c),(x0,y0,x1,y1),(x0-208+50-198*c,278+r*2193)))
s31=save('Figure_S31',blocks,(5660,4574),{'horizontal_col1_col2_minimum':[418,220],'horizontal_col2_col3_minimum':[338,140],'rows':[266,140],'legend_to_top_row':[323,130],'outer_left':[208,50],'outer_top':[114,50],'outer_right':[213,50],'outer_bottom':[230,50]})
(W/'COMPOSITION_REPORT.json').write_text(json.dumps([s08,s31],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{k:v[k]for k in ['figure','source_size','output_size','area_reduction_percent','all_nonwhite_source_pixels_preserved']}for v in [s08,s31]],ensure_ascii=False))
