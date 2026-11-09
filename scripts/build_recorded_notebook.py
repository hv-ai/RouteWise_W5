"""Make a CPU-only demonstration from the user's actual completed GPU export."""
import ast
import hashlib
import json
from pathlib import Path
from build_notebook import code,markdown

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/development_results/desk_records_reviewed.json'
records=json.loads(source.read_text())
output=ROOT/'notebooks/03_CPU_Replay_of_Recorded_Results.ipynb'
modules={f'routewise/{name}':(ROOT/'routewise'/name).read_text()
         for name in ['__init__.py','core.py','demo.py','recorded.py']}
cells=[markdown('''
    # RouteWise recorded demonstration without a GPU

    **This notebook replays recorded scores from the completed Colab GPU run. It does not
    run a model or classify new tickets.** The eight underlying requests are synthetic demo
    examples, and their seven-label scores were produced by the actual trained Qwen adapter.

    Select **Runtime → Change runtime type → Hardware accelerator → None (CPU)**,
    or choose **Connect without GPU**. Run the three cells below in order.
    No model download, backup upload, Google Drive mount, or paid compute is required.

    You can demonstrate review thresholds, human decisions, and suspected-incident grouping
    using the recorded examples. Every exported record marks whether inference was executed.
    Performance measurements remain the original development measurements; replay does not
    provide new validation evidence. The live-model notebook is included separately.
'''),code('''
    #@title 1. Install the notebook controls (CPU only)
    import subprocess,sys
    subprocess.check_call([sys.executable,'-m','pip','install','-q','ipywidgets>=8.1,<9'])
    print('CPU replay controls ready. No GPU is used.')
'''),code(
    '#@title 2. Load the included code and recorded GPU scores\n'
    'from pathlib import Path\nimport json,sys\n'
    'package_files='+repr(modules)+'\n'
    'for relative,source in package_files.items():\n'
    '    target=Path("/content")/relative\n'
    '    target.parent.mkdir(parents=True,exist_ok=True)\n'
    '    target.write_text(source,encoding="utf-8")\n'
    'if "/content" not in sys.path: sys.path.insert(0,"/content")\n'
    'recorded_export='+repr(records)+'\n'
    'from routewise.recorded import RecordedRouter\n'
    'recorded_router=RecordedRouter(recorded_export["tickets"])\n'
    'print("Loaded",len(recorded_export["tickets"]),"recorded tickets. No inference executed.")\n'
    'print("Original export SHA256:",'+repr(hashlib.sha256(source.read_bytes()).hexdigest())+')\n'
),code('''
    #@title 3. Open the recorded replay desk
    from routewise.demo import launch_desk
    desk=launch_desk(recorded_router)
'''),markdown('''
    ## Demonstration sequence

    1. Set **Score cutoff** to **0.90**, keep **Top-two gap** at **0.20**, and enable
       the experimental policy.
    2. Click **Replay 8 recorded tickets** once. Expect four automatic routes and four
       requests for review, using the original recorded scores.
    3. Choose **T0008**, select **Computer-Services**, and save the human decision.
       Expect four automatic, three awaiting review, and one human decision.
    4. Scroll to **Incident Radar**: T0003, T0004, and T0005 form a suspected group.
    5. Download desk records before closing the runtime. They explicitly mark recorded mode.

    The editable text box accepts only exact texts from the eight-record collection.
    Other text becomes an explicit review item with no invented score. Use the live notebook
    when GPU access is available for new-ticket inference. Model weights do not change when
    a reviewer confirms a queue. Incident grouping is lexical, not a root-cause diagnosis.
''')]
for i,c in enumerate(cells):
    c['id']=f'routewise-recorded-{i:02d}'
    if c['cell_type']=='code': ast.parse(c['source'])
notebook={'nbformat':4,'nbformat_minor':5,
          'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                      'language_info':{'name':'python'},'colab':{'name':output.name}},'cells':cells}
output.write_text(json.dumps(notebook,indent=1))
print('CPU-only recorded notebook built and Python cells checked:',output.name)
