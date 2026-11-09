"""Build a standalone Colab notebook from the reviewed local modules."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/01_Restore_and_Try_RouteWise.ipynb"
EXPECTED_BACKUP_SHA256 = '19b99c7092b87829509ee57f1f97669374a22128ad5c2857644187255919fdc4'


def code(source):
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {}, "source": textwrap.dedent(source).strip() + '\n'}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(source).strip() + '\n'}


def build():
    modules = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT/'routewise').glob('*.py'))}
    expected_hash = EXPECTED_BACKUP_SHA256
    cells = [markdown('''
        # RouteWise Incident Desk

        Restore Himani's saved Qwen3 LoRA adapter and try the first working desk.
        **No repeat training and no Google Drive mount are needed.**

        1. Choose **Runtime → Change runtime type → T4 GPU**.
        2. Run cells **1–5** below in order. Cell 2 asks for the original 127.4 MB backup ZIP.
        3. Stop after the desk appears. The development evaluation at the bottom is optional.

        This is an experimental demo. Automatic routing is off by default; the slider values
        have not been selected on independent calibration data. Replay examples are synthetic.
        The original notebook and adapter backup remain unchanged.
    '''), code('''
        #@title 1. Install inference and notebook dependencies
        import subprocess, sys
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers==5.8.0", "peft==0.18.1", "accelerate==1.11.0",
            "safetensors>=0.4.3", "scikit-learn>=1.5,<2",
            "pandas>=2.2,<3", "ipywidgets>=8.1,<9", "matplotlib>=3.9,<4",
        ])
        import torch
        assert torch.cuda.is_available(), "Select a T4 GPU runtime and rerun."
        x = torch.randn(32, 32, device="cuda")
        _ = x @ x
        torch.cuda.synchronize()
        del x, _
        print("GPU computation passed:", torch.cuda.get_device_name(0))
        print("PyTorch:", torch.__version__)
        print("Continue to cell 2. If Colab requests a session restart, restart and rerun this cell first.")
    '''), code('''
        #@title 2. Upload your original backup ZIP
        from google.colab import files
        from pathlib import Path
        import hashlib, json, zipfile, sys
        print("Select RouteWise_backup_20261107_065144_745885.zip (the duplicate (1) is also accepted).")
        uploaded = files.upload()
        archives = [Path(name) for name in uploaded if name.lower().endswith('.zip')]
        assert len(archives) == 1, "Upload exactly one backup ZIP."
        archive = archives[0]
        del uploaded
        expected_hash = "EXPECTED_HASH"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert digest == expected_hash, "This differs from the verified original backup. Stop and check the file."
        BACKUP_DIR = Path('/content/routewise_restored')
        BACKUP_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None, "ZIP integrity check failed."
            for item in zipped.infolist():
                # Intermediate optimizer checkpoints are not needed for inference.
                if item.filename.startswith('adapter/checkpoint-'):
                    continue
                if item.filename.endswith(('.pt', '.bin')):
                    continue
                target = (BACKUP_DIR / item.filename).resolve()
                assert target.is_relative_to(BACKUP_DIR.resolve()), "Unsafe archive path."
                zipped.extract(item, BACKUP_DIR)
        ADAPTER_DIR = BACKUP_DIR / 'adapter'
        assert (ADAPTER_DIR / 'adapter_model.safetensors').exists()
        original_metrics = json.loads((BACKUP_DIR / 'metrics.json').read_text())
        print('Backup restored and fingerprint verified.')
        print('Original measured accuracy:', f"{original_metrics['finetuned_accuracy']:.1%}")
    '''.replace('EXPECTED_HASH', expected_hash)),
    code('#@title Install the included RouteWise source files (no extra upload needed)\n'
         'from pathlib import Path\nimport json, sys\n'
         'package_files = ' + repr(modules) + '\n'
         'for relative, source in package_files.items():\n'
         '    target = Path("/content") / relative\n'
         '    target.parent.mkdir(parents=True, exist_ok=True)\n'
         '    target.write_text(source, encoding="utf-8")\n'
         'if "/content" not in sys.path: sys.path.insert(0, "/content")\n'
         'print("RouteWise modules ready.")\n'),
    code('''
        #@title 3. Load your trained model
        from routewise.inference import Router, BASE_MODEL, BASE_REVISION
        print('Downloading/loading the original base model, then attaching your saved LoRA adapter…')
        router = Router.restore(ADAPTER_DIR)
        print('RESTORED:', BASE_MODEL)
        print('Base revision:', BASE_REVISION)
        print('Saved EOS:', router.tokenizer.eos_token)
        print('Ready for the smoke check. No training was performed.')
    '''), code('''
        #@title 4. Check generation and complete-label scoring
        import pandas as pd
        import json
        smoke_cases = [
            ('Please install Adobe Acrobat on my laptop.', 'Software'),
            ('Please create an Active Directory account for a new employee.', 'Active Directory'),
            ('Please retire the old server and remove it from monitoring.', 'EOL'),
            ('Outlook cannot send email from my mailbox.', 'O365'),
            ('Please grant access to the shared network folder.', 'Fileservice'),
            ('The office printer has a paper jam and needs repair.', 'Computer-Services'),
        ]
        smoke_results = []
        for ticket, expected in smoke_cases:
            generated = router.generate_label(ticket)
            scored = router.score(ticket)
            smoke_results.append({
                'ticket': ticket, 'illustrative_expected': expected,
                'generated_label': generated['prediction'],
                'raw_generation': generated['raw_output'],
                'valid_generation': generated['valid'],
                'scored_label': scored['prediction'],
                'top_relative_score': max(scored['scores'].values()),
                'scores': scored['scores'], 'latency_ms': scored['latency_ms'],
            })
        display(pd.DataFrame(smoke_results).drop(columns=['scores']))
        Path('/content/routewise_smoke_results.json').write_text(json.dumps(smoke_results, indent=2))
        print('These six illustrative examples are not a validation benchmark.')
        print('Different predictions from generation and scoring are possible; keep both results.')
        print('If loading failed or every generation is invalid, share the output before continuing.')
    '''), code('''
        #@title 5. Open the RouteWise Incident Desk
        from routewise.demo import launch_desk
        desk = launch_desk(router)
    '''), markdown('''
        ## Try the first demo

        - Enter a ticket and choose **Route ticket**.
        - Automatic routing defaults off. Enable the experimental policy to try score and gap thresholds.
        - Choose a ticket and human queue, then save the human decision. Original model predictions stay visible.
        - Choose **Replay 8 synthetic tickets**. Routing uses your actual model; incident grouping uses words
          and a few documented email aliases. Similarity is not proof of a shared root cause.
        - Download desk records before closing the runtime; session storage is temporary.

        **Pause here and share the smoke-check output and a screenshot of the desk.**
        Neither a high score nor a successful replay establishes production accuracy.
    '''), markdown('''
        ## Optional development experiment

        Run this only after restoration works. It scores all 117 original validation tickets using
        the same prompt and candidate-label procedure with the adapter on and off. Allow several minutes.
        It does not reproduce the original letter-choice baseline; both measurements are retained separately.

        This set has already been inspected, so the sweep is exploratory. Do not choose the best threshold
        here and call it independent test performance. A separate calibration/test experiment comes next.
    '''), code('''
        #@title 6. Optional paired development evaluation (117 tickets, two models)
        from tqdm.auto import tqdm
        from sklearn.metrics import accuracy_score, classification_report
        from routewise.core import LABELS
        validation = pd.read_csv(BACKUP_DIR / 'val_split.csv')
        development_rows = []
        for _, row in tqdm(validation.iterrows(), total=len(validation)):
            tuned = router.score(row['text'])
            with router.model.disable_adapter():
                base = router.score(row['text'])
            development_rows.append({
                'text': row['text'], 'truth': row['label'],
                'prediction': tuned['prediction'], 'scores': tuned['scores'],
                'log_scores': tuned['log_scores'], 'latency_ms': tuned['latency_ms'],
                'base_prediction': base['prediction'], 'base_scores': base['scores'],
                'base_latency_ms': base['latency_ms'],
            })
        truth = [r['truth'] for r in development_rows]
        dev_metrics = {
            'evaluation_status': 'Exploratory on previously inspected validation examples',
            'base_revision': BASE_REVISION, 'score_method': 'Complete label plus EOS log likelihood',
            'baseline_accuracy': accuracy_score(truth, [r['base_prediction'] for r in development_rows]),
            'tuned_accuracy': accuracy_score(truth, [r['prediction'] for r in development_rows]),
            'tuned_report': classification_report(truth, [r['prediction'] for r in development_rows],
                labels=list(LABELS), target_names=list(LABELS), output_dict=True, zero_division=0),
        }
        print('Base accuracy, same scoring:', f"{dev_metrics['baseline_accuracy']:.1%}")
        print('Tuned accuracy, same scoring:', f"{dev_metrics['tuned_accuracy']:.1%}")
        print('Original tuned generation accuracy:', f"{original_metrics['finetuned_accuracy']:.1%}")
    '''), code('''
        #@title 7. Optional coverage experiment and download
        from routewise.core import Policy, policy_metrics
        import matplotlib.pyplot as plt
        import shutil
        assert 'development_rows' in globals(), 'Run optional cell 6 first.'
        sweep = []
        for value in [.2, .3, .4, .5, .6, .7, .8, .9, .95, .99]:
            sweep.append({'threshold':value, **policy_metrics(
                development_rows, Policy(threshold=value, min_margin=.2, auto_enabled=True))})
        sweep_df = pd.DataFrame(sweep)
        display(sweep_df)
        valid = sweep_df.dropna(subset=['automatic_accuracy'])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(valid['coverage'], valid['automatic_accuracy'], 'o-')
        ax.set(xlabel='Fraction automatically routed', ylabel='Accuracy among automatic routes',
               title='Exploratory review-policy tradeoff', xlim=(0,1), ylim=(0,1.03))
        ax.grid(alpha=.25)
        plt.show()
        export_dir = Path('/content/routewise_development_results')
        export_dir.mkdir(exist_ok=True)
        fig.savefig(export_dir / 'exploratory_coverage.png', dpi=150, bbox_inches='tight')
        sweep_df.to_csv(export_dir / 'exploratory_thresholds.csv', index=False)
        (export_dir / 'scored_tickets.json').write_text(json.dumps(development_rows, indent=2))
        (export_dir / 'metrics.json').write_text(json.dumps(dev_metrics, indent=2))
        (export_dir / 'smoke_results.json').write_text(json.dumps(smoke_results, indent=2))
        (export_dir / 'desk_records.json').write_text(json.dumps(
            {'tickets':desk.tickets, 'reviews':desk.reviews}, indent=2))
        files.download(shutil.make_archive(str(export_dir), 'zip', root_dir=export_dir))
    ''')]
    nb = {'nbformat':4, 'nbformat_minor':5,
          'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                      'language_info':{'name':'python'}, 'accelerator':'GPU', 'colab':{'name':NOTEBOOK.name}},
          'cells':cells}
    for i, cell in enumerate(cells):
        cell['id'] = f'routewise-{i:02d}'
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding='utf-8')


def check():
    nb = json.loads(NOTEBOOK.read_text())
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']), filename=f'notebook-cell-{i}')
    embedded = ast.parse(nb['cells'][3]['source'])
    assignment = next(x for x in embedded.body if isinstance(x, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'package_files' for t in x.targets))
    files = ast.literal_eval(assignment.value)
    for name, source in files.items():
        assert (ROOT/name).read_text() == source, f'Embedded module is stale: {name}'
    print(f'Notebook JSON, Python cells, and {len(files)} embedded source files verified.')


if __name__ == '__main__':
    if '--check' not in sys.argv:
        build()
    check()
