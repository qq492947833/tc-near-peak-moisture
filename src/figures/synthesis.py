from .common import *
import math
import xml.etree.ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium
MM=72/25.4
NS='http://www.w3.org/2000/svg'
INK='http://www.inkscape.org/namespaces/inkscape'
ET.register_namespace('',NS)
ET.register_namespace('inkscape',INK)
NS = 'http://www.w3.org/2000/svg'
INK = 'http://www.inkscape.org/namespaces/inkscape'
INK = 'http://www.inkscape.org/namespaces/inkscape'

def tag(n):
    return '{' + NS + '}' + n

def f(v):
    return f'{v:.6f}'.rstrip('0').rstrip('.') if isinstance(v, (float, int)) else str(v)

def attrs(e):
    out = {'id': e['id'], 'fill': e.get('fill') or 'none', 'stroke': e.get('stroke') or 'none', 'opacity': f(e.get('opacity', 1))}
    if e.get('stroke'):
        out.update({'stroke-width': f(e.get('stroke_width_pt', 1) / MM), 'stroke-linejoin': 'round', 'stroke-linecap': 'round'})
    if e.get('dash'):
        out['stroke-dasharray'] = ' '.join((f(x) for x in e['dash'])) if isinstance(e['dash'], list) else '2 1' if e['dash'] == 'dashed' else '.4 1'
    return out

def svg(scene, lang, path):
    W, H = (scene['canvas']['width_mm'], scene['canvas']['height_mm'])
    root = ET.Element(tag('svg'), {'width': f(W) + 'mm', 'height': f(H) + 'mm', 'viewBox': f'0 0 {W} {H}', 'version': '1.1'})
    ET.SubElement(root, tag('title')).text = 'Stage-resolved environmental pathways'
    defs = ET.SubElement(root, tag('defs'))
    for key, g in scene['gradients'].items():
        a = math.radians(g['angle'])
        dx = math.cos(a) * 0.5
        dy = math.sin(a) * 0.5
        el = ET.SubElement(defs, tag('linearGradient'), {'id': key, 'x1': f(0.5 - dx), 'y1': f(0.5 - dy), 'x2': f(0.5 + dx), 'y2': f(0.5 + dy)})
        for pos, col in g['stops']:
            ET.SubElement(el, tag('stop'), {'offset': f(pos), 'stop-color': col})
    groups = {}
    for e in scene['elements']:
        layer = e['layer']
        if layer not in groups:
            groups[layer] = ET.SubElement(root, tag('g'), {'id': layer, '{' + INK + '}groupmode': 'layer', '{' + INK + '}label': layer})
        parent = groups[layer]
        typ = e['type']
        if typ == 'text':
            lines = e['text'][lang].split('\n')
            fs = e['size_pt'] / MM
            leading = fs * 1.15
            at = {'id': e['id'], 'x': f(e['x']), 'y': f(e['y']), 'font-family': scene['fonts'][lang], 'font-size': f(fs), 'font-weight': 'bold' if e.get('bold') else 'normal', 'fill': e['color'], 'text-anchor': {'left': 'start', 'center': 'middle', 'right': 'end'}[e.get('ha', 'left')]}
            t = ET.SubElement(parent, tag('text'), at)
            for i, line in enumerate(lines):
                ET.SubElement(t, tag('tspan'), {'x': f(e['x']), 'y': f(e['y'] + (i - (len(lines) - 1) / 2) * leading + fs * 0.32)}).text = line
        elif typ == 'path':
            a = attrs(e)
            a['d'] = ' '.join((cmd[0] + (' ' + ' '.join((f(v) for v in cmd[1:])) if len(cmd) > 1 else '') for cmd in e['d']))
            a['fill-rule'] = 'nonzero'
            ET.SubElement(parent, tag('path'), a)
        elif typ == 'ellipse':
            a = attrs(e)
            a.update({k: f(e[k]) for k in ['cx', 'cy', 'rx', 'ry']})
            ET.SubElement(parent, tag('ellipse'), a)
        elif typ == 'rect':
            a = attrs(e)
            a.update({'x': f(e['x']), 'y': f(e['y']), 'width': f(e['w']), 'height': f(e['h'])})
            ET.SubElement(parent, tag('rect'), a)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

def pdf(scene, lang, out):
    W, H = (scene['canvas']['width_mm'], scene['canvas']['height_mm'])
    from matplotlib import font_manager
    for n, weight in [('DejaVu Sans','normal'),('DejaVu Sans-Bold','bold')]:
        if n not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(n,font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight=weight))))
    c = canvas.Canvas(str(out), pagesize=(W * MM, H * MM), pageCompression=0, invariant=1)
    c.setTitle('Synthesis of stage-resolved environmental pathways')
    c.setAuthor('FLTCCF')
    text_checks = []
    for e in scene['elements']:
        typ = e['type']
        c.saveState()
        if typ == 'text':
            font = scene['fonts'][lang] + ('-Bold' if e.get('bold') else '')
            fs = e['size_pt']
            c.setFont(font, fs)
            c.setFillColor(HexColor(e['color']))
            lines = e['text'][lang].split('\n')
            widths = [pdfmetrics.stringWidth(t, font, fs) / MM for t in lines]
            text_checks.append({'id': e['id'], 'max_width_mm': max(widths), 'box_w': e['box_w'], 'fits': max(widths) <= e['box_w'] + 0.1})
            for i, line in enumerate(lines):
                xx = e['x'] * MM
                yy = (H - e['y']) * MM - fs * 0.32 + ((len(lines) - 1) / 2 - i) * fs * 1.15
                {'left': c.drawString, 'center': c.drawCentredString, 'right': c.drawRightString}[e.get('ha', 'left')](xx, yy, line)
        else:
            p = c.beginPath()
            if typ == 'path':
                for cmd in e['d']:
                    op = cmd[0]
                    v = cmd[1:]
                    if op == 'M':
                        p.moveTo(v[0] * MM, (H - v[1]) * MM)
                    elif op == 'L':
                        p.lineTo(v[0] * MM, (H - v[1]) * MM)
                    elif op == 'C':
                        p.curveTo(v[0] * MM, (H - v[1]) * MM, v[2] * MM, (H - v[3]) * MM, v[4] * MM, (H - v[5]) * MM)
                    elif op == 'Z':
                        p.close()
            elif typ == 'ellipse':
                p.ellipse((e['cx'] - e['rx']) * MM, (H - e['cy'] - e['ry']) * MM, 2 * e['rx'] * MM, 2 * e['ry'] * MM)
            elif typ == 'rect':
                p.rect(e['x'] * MM, (H - e['y'] - e['h']) * MM, e['w'] * MM, e['h'] * MM)
            c.setFillAlpha(e.get('opacity', 1))
            c.setStrokeAlpha(e.get('opacity', 1))
            fill = e.get('fill')
            stroke = e.get('stroke')
            if fill and fill.startswith('url(#'):
                g = scene['gradients'][fill[5:-1]]
                x, y, w, h = e['bbox']
                a = math.radians(g['angle'])
                xa = x + w * (0.5 - 0.5 * math.cos(a))
                xb = x + w * (0.5 + 0.5 * math.cos(a))
                ya = y + h * (0.5 - 0.5 * math.sin(a))
                yb = y + h * (0.5 + 0.5 * math.sin(a))
                c.saveState()
                c.clipPath(p, stroke=0, fill=0, fillMode=1)
                c.linearGradient(xa * MM, (H - ya) * MM, xb * MM, (H - yb) * MM, [HexColor(st[1]) for st in g['stops']], positions=[st[0] for st in g['stops']], extend=True)
                c.restoreState()
            elif fill:
                c.setFillColor(HexColor(fill))
                c.setFillAlpha(e.get('opacity', 1))
                c.drawPath(p, stroke=0, fill=1, fillMode=1)
            if stroke:
                c.setStrokeColor(HexColor(stroke))
                c.setStrokeAlpha(e.get('opacity', 1))
                c.setLineWidth(e.get('stroke_width_pt', 1))
                c.setLineJoin(1)
                c.setLineCap(1)
                if e.get('dash'):
                    c.setDash([x * MM for x in e['dash']] if isinstance(e['dash'], list) else [2 * MM, MM] if e['dash'] == 'dashed' else [0.4 * MM, MM])
                c.drawPath(p, stroke=1, fill=0)
        c.restoreState()
    c.showPage()
    c.save()
    bad = [x for x in text_checks if not x['fits']]
    if bad:
        print('TEXT_WIDTH_WARN ' + json.dumps(bad, ensure_ascii=False))
    return text_checks
def render():
 scene=json.loads((DATA/'synthesis/scene.json').read_text(encoding='utf-8'))
 OUTPUT.mkdir(parents=True,exist_ok=True)
 svg(scene,'en',OUTPUT/'Fig9.svg')
 pdf(scene,'en',OUTPUT/'Fig9.pdf')
 with pdfium.PdfDocument(OUTPUT/'Fig9.pdf') as doc:
  page=doc[0]
  bitmap=page.render(scale=DPI/72)
  im=bitmap.to_pil().convert('RGB')
  im.save(OUTPUT/'Fig9.png',dpi=(DPI,DPI))
  im.save(OUTPUT/'Fig9.tiff',dpi=(DPI,DPI),compression='tiff_lzw')
  bitmap.close();page.close()
 MANIFEST.append({'figure':'Fig9','size_mm':[scene['canvas']['width_mm'],scene['canvas']['height_mm']],'dpi':DPI,'editable_elements':len(scene['elements'])})
 print('Rendered Fig9',flush=True)

