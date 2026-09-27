"""Validate source lineage, executable notebook parity, links and bundle hygiene."""
import ast
import hashlib
import json
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
lineage = json.loads((ROOT/'reports/metrics/source_lineage.json').read_text())
for filename, record in lineage.items():
    for folder, field in [('src', 'packaged_sha256'), ('archive/executed', 'executed_sha256')]:
        path = ROOT/folder/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record[field], path
        ast.parse(path.read_text(encoding='utf-8'))
mapping = {'01_build_features.ipynb':'build_features.py', '02_augment_features.ipynb':'augment_features.py',
           '03_train_gpu.ipynb':'train_research.py'}
for filename, script in mapping.items():
    nb = nbformat.read(ROOT/'notebooks'/filename, as_version=4)
    nbformat.validate(nb)
    joined = ''.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert joined == (ROOT/'src'/script).read_text(encoding='utf-8'), filename
    assert all(c.execution_count is None for c in nb.cells if c.cell_type == 'code')
nbformat.validate(nbformat.read(ROOT/'notebooks/00_research.ipynb', as_version=4))
for path in ROOT.rglob('*.md'):
    text = path.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in link or link.startswith('#'):
            continue
        target = link.split('#')[0]
        assert (path.parent/target).exists(), (path, link)
for path in ROOT.rglob('*'):
    if not path.is_file():
        continue
    assert path.name not in {'kaggle.json', 'access_token'}, path
    assert path.suffix not in {'.parquet','.npz','.npy','.ubj','.cbm','.pkl'}, path
    assert path.stat().st_size < 10_000_000, path
    if path.suffix in {'.json','.py','.md','.ipynb','.yml','.txt'}:
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'KGAT_[A-Za-z0-9_]{12,}|^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----$', text, re.M), path
print('Source hashes, notebook parity, Markdown links and artifact allowlist: PASS')
