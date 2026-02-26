"""
parquet_to_pysaliency.py
========================
Optimized for HPC: Converts Parquet scanpaths to PySaliency objects.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pysaliency
from tqdm import tqdm

def load_parquet(parquet_path: Path, image_base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise RuntimeError(f"Parquet file is empty: {parquet_path}")
    if image_base_dir is not None:
        df = df.copy()
        paths = df["image_path"].str.split('/')
        df["image_path"] = str(image_base_dir) + "/" + paths.str[-2] + "/" + paths.str[-1]
    return df

def split_images(df: pd.DataFrame, train_frac: float, seed: int, class_frac: float = 1.0):
    rng = np.random.default_rng(seed)
    class_series = df["image_path"].str.split('/').str[-2]
    all_classes = sorted(class_series.unique())
    n_classes = max(1, int(len(all_classes) * class_frac))
    chosen_classes = set(rng.permutation(all_classes)[:n_classes])
    df = df[class_series.isin(chosen_classes)].copy()
    image_paths = sorted(df["image_path"].unique())
    idx = rng.permutation(len(image_paths))
    n_train = max(1, int(len(image_paths) * train_frac))
    train_imgs = [image_paths[i] for i in idx[:n_train]]
    val_imgs = [image_paths[i] for i in idx[n_train:]] or train_imgs  
    return df, train_imgs, val_imgs

def build_pysaliency(df: pd.DataFrame, img_list: list[str], target_size: int = 224):
    img_index = {p: i for i, p in enumerate(img_list)}
    subset = df[df["image_path"].isin(img_index)]
    train_xs, train_ys, train_ts, train_ns, train_subjects = [], [], [], [], []
    records = subset.to_dict('records')

    for row in tqdm(records, total=len(records), desc="building fixations"):
        n = img_index[row["image_path"]]
        locs = np.array(row["locations"], dtype=np.float32).reshape(-1, 2)
        xs = ((locs[:, 0] + 1.0) * 0.5 * target_size).clip(0, target_size - 1)
        ys = ((locs[:, 1] + 1.0) * 0.5 * target_size).clip(0, target_size - 1)
        ts = np.arange(len(xs), dtype=np.float32)

        train_xs.append(xs)
        train_ys.append(ys)
        train_ts.append(ts)
        train_ns.append(n)
        train_subjects.append(int(row["epoch"]))

    # shapes=(H, W, C) for ImageNet
    shapes = [(target_size, target_size, 3) for _ in img_list]
    stimuli = pysaliency.FileStimuli(filenames=img_list, shapes=shapes)
    
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        xs=train_xs, ys=train_ys, ts=train_ts, ns=train_ns, subjects=train_subjects,
    )
    return stimuli, fixations

def parquet_to_pysaliency(
    parquet_path: Path, train_frac: float = 0.9,
    class_frac: float = 1.0, seed: int = 3141,
    image_base_dir: Path | None = None, target_size: int = 224,
):
    df = load_parquet(Path(parquet_path), image_base_dir)
    df, train_imgs, val_imgs = split_images(df, train_frac, seed, class_frac)
    train_stimuli, train_fixations = build_pysaliency(df, train_imgs, target_size=target_size)
    val_stimuli,   val_fixations   = build_pysaliency(df, val_imgs, target_size=target_size)
    return train_stimuli, train_fixations, val_stimuli, val_fixations