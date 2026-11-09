"""Package only project sources and evidence; never include local environments or weights."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'RouteWise_submission.zip'
ALLOWED={'routewise','scripts','tests','notebooks','data','assets','docs'}
with zipfile.ZipFile(OUTPUT,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if not path.is_file() or '__pycache__' in relative.parts: continue
        if len(relative.parts)==1:
            keep=path.name in {'README.md','.gitignore','requirements-colab.txt'}
        else:
            keep=relative.parts[0] in ALLOWED
        if keep and path.suffix not in {'.zip','.pyc','.pt','.bin','.safetensors'}:
            archive.write(path,Path('RouteWise')/relative)
with zipfile.ZipFile(OUTPUT) as archive:
    assert archive.testzip() is None
    assert not any('/.venv/' in n for n in archive.namelist())
    print('Submission source and evidence ZIP verified:',OUTPUT)
    print('Files:',len(archive.namelist()),'MB:',round(OUTPUT.stat().st_size/1024**2,2))
