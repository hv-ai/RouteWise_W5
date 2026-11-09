"""A fixed TF-IDF comparator on the original 468/117 assignment split."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

LABELS = ['Support general','Fileservice','O365','EOL','Software','Active Directory','Computer-Services']


def main(backup_path, output):
    with zipfile.ZipFile(backup_path) as zipped:
        training = json.loads(zipped.read('TRAIN.json'))
        validation = list(csv.DictReader(io.StringIO(zipped.read('val_split.csv').decode())))
    train_text, train_labels = [], []
    for row in training:
        user = next(m['content'] for m in row['messages'] if m['role']=='user')
        label = next(m['content'] for m in row['messages'] if m['role']=='assistant')
        assert user.startswith('Support Ticket: ')
        train_text.append(user.removeprefix('Support Ticket: ')); train_labels.append(label)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True),
                          LogisticRegression(C=1.0,max_iter=2000,random_state=42))
    model.fit(train_text, train_labels)
    predicted = model.predict([r['text'] for r in validation]).tolist()
    truth = [r['label'] for r in validation]
    results = {
        'status':'Fixed comparator on the original, previously inspected assignment split',
        'train_count':len(training), 'validation_count':len(validation),
        'model':'Word TF-IDF unigrams/bigrams + logistic regression',
        'parameters':{'sublinear_tf':True,'C':1.0,'max_iter':2000,'random_state':42},
        'accuracy':accuracy_score(truth,predicted),
        'report':classification_report(truth,predicted,labels=LABELS,target_names=LABELS,
                                      output_dict=True,zero_division=0),
        'scikit_learn_version':sklearn.__version__, 'python_version':platform.python_version(),
        'backup_sha256':hashlib.sha256(Path(backup_path).read_bytes()).hexdigest(),
        'no_validation_based_hyperparameter_search':True,
    }
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'tfidf_baseline.json').write_text(json.dumps(results,indent=2))
    (output/'tfidf_predictions.json').write_text(json.dumps([
        {'text':r['text'],'truth':r['label'],'prediction':p} for r,p in zip(validation,predicted)
    ],indent=2))
    print(json.dumps({'accuracy':results['accuracy'],'macro_f1':results['report']['macro avg']['f1-score'],
                      'train_count':len(training),'validation_count':len(validation)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backup',required=True)
    parser.add_argument('--output',default='data/development_results')
    args=parser.parse_args()
    main(args.backup,args.output)
