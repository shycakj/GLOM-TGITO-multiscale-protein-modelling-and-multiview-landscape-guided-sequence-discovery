"""Regenerate the five main and fourteen supplementary figures from frozen data."""
from pathlib import Path
import os, sys, subprocess, json, shutil

ROOT=Path(__file__).resolve().parents[1]
PAPER_SOURCE=ROOT/'paper'
PAPER=ROOT/'reproduced/figure_work'
OUT=ROOT/'reproduced/figures';OUT.mkdir(parents=True,exist_ok=True)
LOG=ROOT/'reproduced/logs';LOG.mkdir(parents=True,exist_ok=True)

def run(name,script,*args,code=None):
    path=Path(script);env=os.environ.copy()
    env['PYTHONPATH']=str(path.parent)+os.pathsep+env.get('PYTHONPATH','')
    command=[sys.executable,'-X','utf8']+(['-c',code] if code else [str(path),*args])
    with (LOG/(name+'.log')).open('wb') as f:
        p=subprocess.run(command,cwd=path.parent,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{name}: inspect {LOG/(name+".log")}')
    print('Rendered:',name,flush=True)

def setup():
    # Render only in a working copy; the distributed inputs remain hash-stable.
    PAPER.mkdir(parents=True,exist_ok=True)
    for name in ['data','figure_reproduction','support_plotting','latex']:
        shutil.copytree(PAPER_SOURCE/name,PAPER/name,dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['figure_edit_plan.json','figure_secondary_edit_plan.json']:
        shutil.copy2(PAPER_SOURCE/name,PAPER/name)
    for root in [PAPER/'support_plotting',PAPER/'data/Figure_Source_Data/refinement_20261025']:
        for d in ['figures','qa','latex/bioinformatics/figures','latex/supplement/figures']:(root/d).mkdir(parents=True,exist_ok=True)
    p=PAPER/'support_plotting/figures/figS6_endpoint_gates.caption.txt'
    if not p.exists():p.write_text('Endpoint source-weight evidence.\n')

def clean_pdf(source,destination,plan):
    import fitz
    doc=fitz.open(source);page=doc[0]
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    norm=lambda t:' '.join(t.split())
    removed=[]
    for s in spans:
        if any(norm(s['text']).startswith(norm(v)) for v in plan.get('remove',[])) or s['text'] in plan.get('remove_exact',[]):
            box=fitz.Rect(s['bbox']);y=s['origin'][1]
            page.add_redact_annot(fitz.Rect(box.x0-.2,y-.45*s['size'],box.x1+.2,y-.25*s['size']),fill=None,cross_out=False)
            removed.append(s['text'])
    # Keep data marks and molecular images intact.
    page.apply_redactions(images=0,graphics=0,text=0)
    for old,new in plan.get('replace',{}).items():
        matches=[s for s in spans if norm(s['text'])==norm(old)]
        for s in matches:
            box=fitz.Rect(s['bbox']);y=s['origin'][1]
            page.add_redact_annot(fitz.Rect(box.x0-.2,y-.45*s['size'],box.x1+.2,y-.25*s['size']),fill=None,cross_out=False)
            page.apply_redactions(images=0,graphics=0,text=0)
            import matplotlib
            fontpath=Path(matplotlib.get_data_path())/'fonts/ttf'/('DejaVuSans-Bold.ttf' if 'Bold' in s['font'] else 'DejaVuSans.ttf')
            # Prefer the original font where all required glyphs are available.
            fonts=[f for f in page.get_fonts(full=True) if f[3].split('+')[-1].replace('-','')==s['font'].replace('-','')]
            fb=doc.extract_font(fonts[0][0])[3] if fonts else fontpath.read_bytes()
            font=fitz.Font(fontbuffer=fb)
            if not all(font.has_glyph(ord(c)) for c in new if not c.isspace()):fb=fontpath.read_bytes();font=fitz.Font(fontbuffer=fb)
            key='Label'+str(len(removed));page.insert_font(fontname=key,fontbuffer=fb)
            x=(box.x0+box.x1)/2-font.text_length(new,fontsize=s['size'])/2
            rgb=((s['color']>>16&255)/255,(s['color']>>8&255)/255,(s['color']&255)/255)
            page.insert_text((x,y),new,fontsize=s['size'],fontname=key,color=rgb)
            removed.append(old+' -> '+new)
    crop=plan.get('crop')
    if crop:
        rect=fitz.Rect(crop);rect.x1=min(rect.x1,page.mediabox.x1);rect.y1=min(rect.y1,page.mediabox.y1)
        # Figure 4 was regenerated without the subsequently removed 40-pt header.
        if source.name=='fig4_integrated_evidence.pdf':rect=page.rect
        page.set_cropbox(rect)
    if source.name=='fig2_keysite_example.pdf':
        # Carry forward the final, approved single coordinate label.
        for label in ['UMAP 1','UMAP 2']:
            for box in page.search_for(label):page.add_redact_annot(box+(-.8,-.8,.8,.8),fill=(1,1,1))
        page.apply_redactions(images=0,graphics=0,text=0)
        fontpath=Path('C:/Windows/Fonts/arial.ttf');key='AxisArial' if fontpath.exists() else 'helv'
        if fontpath.exists():page.insert_font(fontname=key,fontfile=str(fontpath))
        ret=page.insert_textbox(fitz.Rect(73,231.5,174,242.4),'Landscape coordinates',fontname=key,
            fontsize=7.3,color=(36/255,55/255,70/255),align=1)
        assert ret>=0
    if source.name=='fig4_integrated_evidence.pdf':
        # The approved crop retains two points of whitespace above the compact panel.
        padded=fitz.open();newpage=padded.new_page(width=page.rect.width,height=page.rect.height+2)
        newpage.show_pdf_page(fitz.Rect(0,2,page.rect.width,page.rect.height+2),doc,0)
        doc.close();doc=padded;page=doc[0]
    doc.save(destination,garbage=4,deflate=True);doc.close()
    with fitz.open(destination) as d:d[0].get_pixmap(dpi=250,alpha=False).save(destination.with_suffix('.png'))
    return removed

def main():
    setup()
    support=PAPER/'support_plotting/source'
    if '--assemble-only' not in sys.argv:
        new=PAPER/'figure_reproduction/source'
        run('main1',new/'reproduce_architecture.py')
        run('main5_S13_S14',new/'reproduce_figures.py')
        refined=PAPER/'data/Figure_Source_Data/refinement_20261025/source'
        run('main2_main3_S3_S10',refined/'render_final_figures.py','terrain','performance')
        for name in ['render_structure_plate','render_discovery','render_supplement_evidence','render_tgito_mechanism','render_protocol']:
            run(name,support/(name+'.py'))
        run('S2_S11',support/'render_updated_figures.py',code='import render_updated_figures as r; r.supplement_prediction(); r.saprot_metrics()')
        run('S12',support/'render_selected_figures.py',code='import render_selected_figures as r; r.supplement()')
    run('S6',support/'render_terrain_journal_supplement.py',code='import render_terrain_journal_supplement as r; r.endpoints()')
    run('main4',support/'render_integrated_figures.py',code='import render_integrated_figures as r; r.discovery()')
    plan=json.loads((PAPER/'figure_edit_plan.json').read_text(encoding='utf8'))
    secondary=json.loads((PAPER/'figure_secondary_edit_plan.json').read_text(encoding='utf8'))
    locations={}
    for root in [PAPER/'support_plotting/figures',PAPER/'data/Figure_Source_Data/refinement_20261025/figures',PAPER/'figure_reproduction/figures',PAPER/'figure_reproduction/latex/bioinformatics/figures']:
        for p in root.glob('*.pdf'):locations[p.name]=p
    audit=[]
    for section in ['bioinformatics','supplement']:
        target=OUT/section;target.mkdir(exist_ok=True)
        for f in (PAPER/'latex'/section/'figures').glob('*.pdf'):
            if f.name not in locations:raise FileNotFoundError('No regenerated figure: '+f.name)
            source=locations[f.name]
            removed=clean_pdf(source,target/f.name,plan.get(f.name,{}) if f.name not in {'fig1_integrated.pdf'} else {})
            if f.name in secondary:
                temporary=target/(f.stem+'_stage1.pdf');(target/f.name).replace(temporary)
                removed+=clean_pdf(temporary,target/f.name,secondary[f.name])
                temporary.unlink()
            audit.append({'figure':f.name,'source':source.relative_to(ROOT).as_posix(),'removed_annotation_spans':removed})
    assert len(audit)==19
    (OUT/'REPRODUCTION.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
    print('PASS: all 19 publication figures regenerated in',OUT)

if __name__=='__main__':main()
