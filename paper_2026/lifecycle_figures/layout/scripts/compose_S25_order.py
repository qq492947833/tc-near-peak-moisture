"""Move the two native row groups of supplementary figure 25, preserving all marks."""
from pathlib import Path
import argparse,json,hashlib
from PIL import Image,ImageChops

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def white(image):return image.getextrema()==((255,255),)*3
def compose(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if digest(source)!="536219a0cd443f6278cdccd5ada4b45b8283eeef468cf4a4999763ced390c607":raise ValueError("Expected the accepted pre-reorder moisture panel")
    im=Image.open(source).convert('RGB')
    assert im.size==(6377,5527)
    starts=[0,2674];glyphs=[[(298,114,373,193),(3474,88,3554,193)],[(298,114,364,193),(3469,88,3549,193)]]
    rows=[];labels=[];canvas=im.copy()
    for i,y in enumerate(starts):
        box=(0,y,im.width,y+2610)
        row=im.crop(box)
        labels.append([row.crop(g) for g in glyphs[i]])
        for g,letter in zip(glyphs[i],labels[-1]):
            assert all(white(letter.crop(b)) for b in [(0,0,1,letter.height),(0,0,letter.width,1),(letter.width-1,0,letter.width,letter.height),(0,letter.height-1,letter.width,letter.height)])
            row.paste((255,255,255),g)
        rows.append(row);canvas.paste((255,255,255),box)
    assert white(canvas)
    for i in range(2):
        row=rows[1-i].copy()
        for g,letter in zip(glyphs[i],labels[i]):
            assert white(row.crop(g));row.paste(letter,g[:2])
        y=starts[i];canvas.paste(row,(0,y));check=canvas.crop((0,y,im.width,y+2610))
        for g in glyphs[i]:check.paste((255,255,255),g)
        assert ImageChops.difference(rows[1-i],check).getbbox() is None
    canvas.save(out/'Figure_S25.png',dpi=(900,900));canvas.save(out/'Figure_S25.tiff',dpi=(900,900),compression='tiff_lzw')
    assert ImageChops.difference(canvas,Image.open(out/'Figure_S25.tiff').convert('RGB')).getbbox() is None
    return {'source_sha256':digest(source),'new_by_old':'cdab','native_pixels':im.size,'unchanged_scientific_pixels':True,'sha256':{ext:digest(out/('Figure_S25.'+ext)) for ext in ('png','tiff')}}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(compose(a.source,a.out),indent=2))
