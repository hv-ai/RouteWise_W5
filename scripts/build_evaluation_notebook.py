"""Build the prospective split/calibration experiment as a standalone Colab file."""
import ast
import json
from pathlib import Path
from build_notebook import code, markdown, build, EXPECTED_BACKUP_SHA256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'notebooks/02_Train_Calibrate_and_Test_RouteWise.ipynb'
build()  # Refresh embedded modules in notebook 01 as well.
first = json.loads((ROOT/'notebooks/01_Restore_and_Try_RouteWise.ipynb').read_text())
cells = [markdown('''
    # RouteWise separate training calibration and test experiment

    The desk already works. This experiment strengthens the evidence for deciding which
    tickets to route automatically. It trains a **new adapter from the original base model**.
    It does not overwrite the original model or change the original assignment results.

    Use a **fresh T4 Colab runtime**. Run the cells in order. The original backup ZIP is
    uploaded again; Google Drive is not required. Keep the original desk notebook saved.

    **Protocol fixed before this run:** seed 20261108; stratified 60/20/20 split; 3 training
    epochs; LoRA rank 8; learning rate 5e-5. Only training examples update model weights.
    Choose a review policy on calibration examples, aiming for at least 90% observed
    automatic-route accuracy with at least 20 accepted examples. If none qualifies,
    route everything to review. This is a selection rule, not an accuracy guarantee.
    Evaluate the frozen policy on the test partition once, including sample counts and
    an approximate 95% Wilson interval. Fit a simple TF-IDF classifier on training data.

    The source dataset and its earlier results have been inspected already. This is a new
    model-held-out split of the same dataset, not an independent collection of real tickets.
    Do not repeatedly change settings after viewing test results. Synthetic replay remains
    a separate demonstration and is not benchmark evidence.
''')]
cells.extend(first['cells'][1:4])  # Dependencies, verified upload, embedded source.
cells += [code('''
    #@title A. Prepare and save the three separate partitions
    from routewise.experiment import recover_dataset, split_dataset
    from routewise.core import LABELS, Policy, policy_metrics
    from routewise.inference import Router, BASE_MODEL, BASE_REVISION
    from pathlib import Path
    import pandas as pd
    import json, hashlib, gc
    SEED = 20261108
    RUN_DIR = Path('/content/routewise_separate_evaluation')
    RUN_DIR.mkdir(exist_ok=True)
    partitions = split_dataset(recover_dataset(BACKUP_DIR), seed=SEED)
    assert [len(partitions[k]) for k in ['train','calibration','test']] == [351,117,117]
    for name, rows in partitions.items():
        pd.DataFrame(rows).to_csv(RUN_DIR / f'{name}_split.csv', index=False)
    manifest = {
        'seed':SEED, 'base_model':BASE_MODEL, 'base_revision':BASE_REVISION,
        'original_adapter_used_for_training':False,
        'prior_dataset_exposure':'Original results and examples already inspected; not external validation.',
        'split_ids':{k:[r['id'] for r in v] for k,v in partitions.items()},
        'counts':{k:len(v) for k,v in partitions.items()},
        'protocol':{'epochs':3, 'learning_rate':5e-5, 'lora_rank':8,
                    'calibration_target':.90, 'calibration_min_automated':20},
    }
    (RUN_DIR/'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    display(pd.DataFrame({name:pd.Series([r['label'] for r in rows]).value_counts()
                          for name,rows in partitions.items()}).fillna(0).astype(int))
    print('Disjoint by normalized text ID. Counts: 351 train / 117 calibration / 117 test.')
    print('Near-duplicate meaning is not automatically detected by this check.')
'''), code('''
    #@title B. Fit the simple text-classification comparator (training examples only)
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    classical = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    )
    classical.fit([r['text'] for r in partitions['train']],
                  [r['label'] for r in partitions['train']])
    print('TF-IDF + logistic regression fitted. Test predictions will be measured after policy freeze.')
'''), code('''
    #@title C. Train a separate adapter from the base model (one new training run)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    set_seed(SEED)
    NEW_ADAPTER = RUN_DIR/'adapter'
    assert not (NEW_ADAPTER/'adapter_config.json').exists(), (
        'A trained extension adapter is already saved here. Do not retrain to chase test results. '
        'If this cell completed previously, continue to calibration.')
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, torch_dtype=torch.float16,
        attn_implementation='sdpa',
    )
    base.config.use_cache = False
    base.config.eos_token_id = tokenizer.eos_token_id
    base.config.pad_token_id = tokenizer.pad_token_id
    train_model = get_peft_model(base, LoraConfig(
        task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    ))
    formatter = Router(train_model, tokenizer)
    examples = []
    for row in partitions['train']:
        prompt_ids = formatter._prompt_ids(row['text'])
        answer_ids = formatter.candidates[row['label']]
        examples.append({
            'input_ids':prompt_ids+answer_ids,
            'labels':[-100]*len(prompt_ids)+answer_ids,
        })
    def collate(batch):
        length = max(len(row['input_ids']) for row in batch)
        return {
            'input_ids':torch.tensor([row['input_ids']+[tokenizer.pad_token_id]*(length-len(row['input_ids'])) for row in batch]),
            'attention_mask':torch.tensor([[1]*len(row['input_ids'])+[0]*(length-len(row['input_ids'])) for row in batch]),
            'labels':torch.tensor([row['labels']+[-100]*(length-len(row['labels'])) for row in batch]),
        }
    arguments = TrainingArguments(
        output_dir=str(RUN_DIR/'training'), num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=5e-5, lr_scheduler_type='cosine', warmup_steps=0,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=5, save_strategy='no', report_to='none',
        seed=SEED, data_seed=SEED, dataloader_num_workers=0,
    )
    trainer = Trainer(model=train_model, args=arguments, train_dataset=examples,
                      data_collator=collate, processing_class=tokenizer)
    train_model.train()
    trained = trainer.train()
    trainer.save_model(str(NEW_ADAPTER))
    tokenizer.save_pretrained(str(NEW_ADAPTER))
    trainer.save_state()
    (RUN_DIR/'training_metrics.json').write_text(json.dumps(trained.metrics, indent=2))
    (RUN_DIR/'training_log.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    train_model.gradient_checkpointing_disable()
    train_model.config.use_cache = True
    train_model.generation_config.eos_token_id = tokenizer.eos_token_id
    train_model.generation_config.pad_token_id = tokenizer.pad_token_id
    evaluation_router = Router(train_model, tokenizer)
    print('New adapter trained and saved separately. Original adapter has not been modified.')
'''), code('''
    #@title D. Choose and freeze the review policy using calibration examples only
    from routewise.experiment import choose_policy
    from tqdm.auto import tqdm
    from datetime import datetime, timezone
    assert not (RUN_DIR/'test_results.json').exists(), 'Test results already exist. Do not retune after viewing them.'
    calibration_rows = []
    for row in tqdm(partitions['calibration'], desc='Calibration scoring'):
        result = evaluation_router.score(row['text'])
        calibration_rows.append({'id':row['id'], 'truth':row['label'], **result})
    choice = choose_policy(calibration_rows, target_accuracy=.90, min_automated=20)
    choice['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    weights = NEW_ADAPTER/'adapter_model.safetensors'
    choice['adapter_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    choice['split_manifest_sha256'] = hashlib.sha256((RUN_DIR/'split_manifest.json').read_bytes()).hexdigest()
    frozen_text = json.dumps(choice, indent=2)
    (RUN_DIR/'frozen_policy.json').write_text(frozen_text)
    frozen_policy_hash = hashlib.sha256(frozen_text.encode()).hexdigest()
    (RUN_DIR/'frozen_policy.sha256').write_text(frozen_policy_hash)
    (RUN_DIR/'calibration_scores.json').write_text(json.dumps(calibration_rows, indent=2))
    selected_policy = Policy(**choice['policy'])
    print('FROZEN POLICY:', choice['policy'])
    print('Calibration performance:', choice['calibration_metrics'])
    if choice['fallback_review_all']:
        print('No candidate met the predeclared calibration criteria: all tickets will go to review.')
    print('Policy selection is complete. Test examples have not been scored in this run.')
'''), code('''
    #@title E. Evaluate the frozen policy and comparators on test examples once
    from sklearn.metrics import accuracy_score, classification_report
    from routewise.experiment import wilson_interval
    assert hashlib.sha256((RUN_DIR/'frozen_policy.json').read_bytes()).hexdigest() == frozen_policy_hash
    assert hashlib.sha256((NEW_ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest() == choice['adapter_sha256']
    test_rows = []
    for row in tqdm(partitions['test'], desc='Final test scoring'):
        tuned = evaluation_router.score(row['text'])
        with evaluation_router.model.disable_adapter():
            baseline = evaluation_router.score(row['text'])
        test_rows.append({
            'id':row['id'], 'truth':row['label'], 'text':row['text'],
            'prediction':tuned['prediction'], 'scores':tuned['scores'],
            'log_scores':tuned['log_scores'], 'latency_ms':tuned['latency_ms'],
            'base_prediction':baseline['prediction'], 'base_latency_ms':baseline['latency_ms'],
        })
    truth = [r['truth'] for r in test_rows]
    simple_predictions = classical.predict([r['text'] for r in partitions['test']]).tolist()
    for row, prediction in zip(test_rows, simple_predictions): row['tfidf_prediction'] = prediction
    selective = policy_metrics(test_rows, selected_policy)
    interval = wilson_interval(selective['automated']-selective['automatic_errors'], selective['automated'])
    results = {
        'status':'New model-held-out split; source dataset previously inspected, not external validation.',
        'test_count':len(test_rows), 'policy':choice['policy'],
        'policy_sha256':frozen_policy_hash, 'policy_selected_on':'calibration',
        'base_accuracy':accuracy_score(truth, [r['base_prediction'] for r in test_rows]),
        'tuned_accuracy':accuracy_score(truth, [r['prediction'] for r in test_rows]),
        'tfidf_accuracy':accuracy_score(truth, simple_predictions),
        'selective_routing':selective, 'automatic_accuracy_wilson95':interval,
        'tuned_report':classification_report(truth, [r['prediction'] for r in test_rows],
            labels=list(LABELS), target_names=list(LABELS), output_dict=True, zero_division=0),
    }
    (RUN_DIR/'test_results.json').write_text(json.dumps(results, indent=2))
    (RUN_DIR/'test_predictions.json').write_text(json.dumps(test_rows, indent=2))
    display(pd.DataFrame([
        {'Model':'Base Qwen, full-label scoring', 'Test accuracy':results['base_accuracy']},
        {'Model':'New tuned Qwen, full-label scoring', 'Test accuracy':results['tuned_accuracy']},
        {'Model':'TF-IDF + logistic regression', 'Test accuracy':results['tfidf_accuracy']},
    ]))
    print('Frozen-policy test results:', selective)
    print('Approximate 95% interval for automatic-route accuracy:', interval)
    print('Report every comparator, even if the simpler model wins. Do not retune using test results.')
'''), code('''
    #@title F. Download the new adapter and experiment evidence
    import zipfile
    from google.colab import files
    archive = Path('/content/RouteWise_separate_evaluation.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(RUN_DIR.rglob('*')):
            if path.is_file(): zipped.write(path, path.relative_to(RUN_DIR))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert 'test_results.json' in zipped.namelist()
        assert 'adapter/adapter_model.safetensors' in zipped.namelist()
    print('Archive verified. Download it, then share it in the RouteWise chat.')
    files.download(str(archive))
''')]

for i, cell in enumerate(cells):
    cell['id'] = f'routewise-evaluation-{i:02d}'
    if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))
nb = {'nbformat':4, 'nbformat_minor':5,
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info':{'name':'python'}, 'accelerator':'GPU', 'colab':{'name':OUTPUT.name}},
      'cells':cells}
OUTPUT.write_text(json.dumps(nb, indent=1))
print('Separate-evaluation notebook built; every Python cell parses.')
