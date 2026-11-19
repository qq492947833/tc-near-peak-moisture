"""Reorder complete native-resolution panels without changing any plotted observation."""
from pathlib import Path
import argparse,hashlib,json
from PIL import Image,ImageChops
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def all_white(im):return im.getextrema()==((255,255),)*3
def compose(number,source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if digest(source)!={5:'df0ece99a50db84881ad814520cd4c36444e9ef39c898c16f79108208b5712b2',6:'c92785521972f46cf369550b74b72aa765c23539db716349c3c8a5480544680b'}[number]:raise ValueError('Incorrect pre-reorder panel image')
    im=Image.open(source).convert('RGB');dpi=900
    if number==5:
        assert im.size==(6377,7015)
        w=3188; starts=(220,2412,4604);heights=(2120,2120,2120);dst=starts
        order='abcfde';glyphs=[(513,383,576,450),(3698,361,3765,450),(513,2575,568,2642),(3693,2553,3761,2642),(513,4767,580,4834),(3691,4745,3740,4832)]
    elif number==6:
        assert im.size==(6377,7157)
        w=3144;starts=(0,1787,3573);heights=(1787,1786,1927);dst=(0,1787,3771)
        order='acdebf';glyphs=[(193,72,266,149),(3338,46,3415,148),(193,1859,257,1936),(3336,1833,3413,1936),(193,3645,271,3722),(3332,3619,3389,3720)]
    else:raise ValueError(number)
    canvas=im.copy();tiles=[];letters=[];boxes=[]
    for k in range(6):
        r,c=divmod(k,2);x=w*c;y=starts[r];box=(x,y,x+w,y+heights[r]);tile=im.crop(box)
        # The standalone panel letters are separated from all titles, axes and marks.
        l,t,rr,b=glyphs[k];l-=1;t-=1;rr+=1;b+=1
        letter=im.crop((l,t,rr,b));letterbox=(l-x,t-y,rr-x,b-y)
        assert all(all_white(letter.crop(bb)) for bb in [(0,0,1,letter.height),(0,0,letter.width,1),(letter.width-1,0,letter.width,letter.height),(0,letter.height-1,letter.width,letter.height)])
        tile.paste((255,255,255),letterbox)
        tiles.append(tile);letters.append((letter,letterbox));boxes.append(box)
        canvas.paste((255,255,255),box)
    baseline=canvas.copy(); manifest=[]
    for k,ch in enumerate(order):
        j=ord(ch)-97;r,c=divmod(k,2);left=c*w;top=dst[r]
        tile=tiles[j].copy();lbox=letters[k][1]
        assert all_white(tile.crop(lbox))
        tile.paste(letters[k][0],lbox[:2])
        dest=(left,top,left+w,top+tile.height)
        assert all_white(canvas.crop(dest))
        canvas.paste(tile,(left,top))
        final=canvas.crop(dest);final.paste((255,255,255),lbox)
        assert ImageChops.difference(tiles[j],final).getbbox() is None
        manifest.append({'old':ch,'new':chr(97+k),'source':boxes[j],'destination':dest,
                         'scientific_pixel_hash':hashlib.sha256(tiles[j].tobytes()).hexdigest()})
    # Recover all unchanged shared objects, including outer labels, legends and Fig. 6g.
    rest=canvas.copy()
    for m in manifest:rest.paste((255,255,255),tuple(m['destination']))
    assert ImageChops.difference(baseline,rest).getbbox() is None
    if number==6:
        # The last time-series axes in the left column are now panel c. Move
        # the existing shared time xlabel from panel d to c at the same height.
        x,y,w_label,h_label=3144,5355-3573+1787,3144,107
        box=(x,y,x+w_label,y+h_label); dest=(0,y,w_label,y+h_label)
        xlabel=canvas.crop(box); assert not all_white(xlabel) and all_white(canvas.crop(dest))
        canvas.paste((255,255,255),box);canvas.paste(xlabel,(0,y))
        assert ImageChops.difference(xlabel,canvas.crop(dest)).getbbox() is None
        manifest[3]['shared_time_xlabel_to']='c'; manifest[3]['shared_time_xlabel_bbox']=box
    stem='Figure_%02d'%number
    canvas.save(out/(stem+'.png'),dpi=(dpi,dpi));canvas.save(out/(stem+'.tiff'),dpi=(dpi,dpi),compression='tiff_lzw')
    with Image.open(out/(stem+'.tiff')) as raster:assert ImageChops.difference(raster.convert('RGB'),canvas).getbbox() is None
    return {'source_sha256':digest(source),'shape':canvas.size,'slot_mapping':manifest,'tiff_png_pixels_equal':True,
            'unchanged_shared_objects':True,'sha256':{ext:digest(out/(stem+'.'+ext)) for ext in ('png','tiff')}}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--number',type=int,choices=[5,6],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compose(a.number,a.source,a.out),indent=2))
