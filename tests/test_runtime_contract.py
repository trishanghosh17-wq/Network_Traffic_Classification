import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

def test_canonical_artifact_contract():
    meta=json.loads((ROOT/'models/model_metadata.json').read_text())
    schema=json.loads((ROOT/'models/feature_schema.json').read_text())
    model=joblib.load(ROOT/'models/model.joblib')
    assert meta['artifact'] == 'models/model.joblib'
    assert meta['features'] == schema['features']
    assert len(schema['features']) == model.n_features_in_
    assert meta['n_features'] == model.n_features_in_

def test_ood_schema_matches_when_enabled():
    meta=json.loads((ROOT/'models/model_metadata.json').read_text())
    ood_meta=json.loads((ROOT/'models/ood_metadata.json').read_text())
    assert ood_meta['features'] == meta['features']
