"""Export measured comparison and importance tables from immutable run results."""
import csv
import json
from pathlib import Path
import validation

ROOT=validation.ROOT;OUT=ROOT/'results/v5_experiment'


def write_table(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    e=json.loads((OUT/'experiment.json').read_text())
    a=json.loads((ROOT/'results/v2_v4_diagnosis/audit.json').read_text())
    rows=[{'model':name,'users':len(e['outer_users']),**v} for name,v in e['outer_comparisons'].items()]
    rows.append({'model':'V4_saved_model','users':len(e['outer_users']),
        'primary_all120':a['paired_metrics']['primary_all_120']['v4_map'],
        'secondary_top5_120':a['paired_metrics']['top5_120']['v4_map'],
        'hypothesis_top5_80':a['paired_metrics']['top5_fraction80']['v4_map']})
    write_table(OUT/'comparison.csv',rows)
    write_table(OUT/'feature_importance.csv',e['feature_importance'])
    print('Exported measured comparisons and feature importance.')


if __name__=='__main__':main()
