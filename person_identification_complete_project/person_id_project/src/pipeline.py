
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


def get_face_app():
    if FaceAnalysis is None:
        raise RuntimeError('insightface is not installed')
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def load_image(path):
    return np.array(Image.open(path).convert('RGB'))


def extract_embeddings(image_paths):
    app = get_face_app()
    rows = []
    for path in tqdm(image_paths, desc='Embedding images'):
        try:
            img = load_image(path)
            faces = app.get(img)
            if not faces:
                rows.append({'path': str(path), 'embedding': None, 'det_score': 0.0, 'bbox': None})
                continue
            face = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
            rows.append({'path': str(path), 'embedding': face.normed_embedding.astype(np.float32), 'det_score': float(getattr(face, 'det_score', 0.0)), 'bbox': list(map(float, face.bbox))})
        except Exception:
            rows.append({'path': str(path), 'embedding': None, 'det_score': 0.0, 'bbox': None})
    return rows


def cluster_embeddings(rows):
    valid_idx = [i for i, r in enumerate(rows) if r['embedding'] is not None]
    if not valid_idx:
        return rows
    X = np.stack([rows[i]['embedding'] for i in valid_idx])
    dist = cosine_distances(X)
    labels = DBSCAN(eps=0.28, min_samples=2, metric='precomputed').fit_predict(dist)
    for i, label in zip(valid_idx, labels):
        rows[i]['cluster'] = int(label) if label >= 0 else -1
    for i in range(len(rows)):
        rows[i].setdefault('cluster', -1)
    return rows


def add_confidence(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'embedding'} for r in rows])
    if df.empty:
        return df
    df['confidence'] = 0.0
    clustered = df[df['cluster'] >= 0]
    if clustered.empty:
        return df
    for c in sorted(clustered['cluster'].unique()):
        idx = [i for i, r in enumerate(rows) if r['cluster'] == c and r['embedding'] is not None]
        group = np.stack([rows[i]['embedding'] for i in idx])
        centroid = group.mean(axis=0, keepdims=True)
        conf = np.clip(1.0 - cosine_distances(group, centroid).ravel(), 0.0, 1.0)
        for j, i in enumerate(idx):
            df.loc[i, 'confidence'] = float(conf[j])
    return df


def make_summary(df):
    return df.groupby('cluster', as_index=False).agg(image_count=('path', 'count'), mean_confidence=('confidence', 'mean')).sort_values(['image_count', 'cluster'], ascending=[False, True])


def save_montage(df, output_dir):
    clusters = [c for c in sorted(df['cluster'].unique()) if c >= 0][:12]
    if not clusters:
        return
    fig, axes = plt.subplots(len(clusters), 4, figsize=(12, 3 * len(clusters)))
    if len(clusters) == 1:
        axes = np.expand_dims(axes, 0)
    for r, c in enumerate(clusters):
        subset = df[df['cluster'] == c].head(4)
        for i in range(4):
            ax = axes[r, i]
            ax.axis('off')
            if i < len(subset):
                row = subset.iloc[i]
                ax.imshow(Image.open(row['path']).convert('RGB'))
                ax.set_title('c' + str(c) + ' conf=' + str(round(float(row['confidence']), 2)))
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'cluster_montage.png', dpi=160, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data')
    parser.add_argument('--output_dir', default='outputs')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    rows = add_confidence(cluster_embeddings(extract_embeddings(image_paths)))
    summary = make_summary(rows)

    rows.to_csv(output_dir / 'clusters.csv', index=False)
    summary.to_csv(output_dir / 'cluster_summary.csv', index=False)
    with open(output_dir / 'clusters.json', 'w') as f:
        json.dump(rows.to_dict(orient='records'), f, indent=2)
    save_montage(rows, output_dir)
    print('Saved outputs to ' + str(output_dir))


if __name__ == '__main__':
    main()
