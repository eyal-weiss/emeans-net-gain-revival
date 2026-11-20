"""Additional small-to-medium datasets (2026-11-19, after the confirmatory study): parsers only; the preparation protocol
is datasets.py's (_finish: remove exact duplicates, hold out 20,000 rows with RandomState(1234), keep the second half of
the held-out rows as the test set, scale with training statistics).

Selection rule (fixed before any run on them): public datasets with dense real-valued features, at least 60,000 unique
rows and at most ~600,000, used in no earlier experiment of this project, spread over dimensions (3 to 784). Scaling:
z-score for tabular data; min-max ("mm") for pixel images, where per-pixel z-scores would turn rarely-lit pixels into
artificial outliers. A dataset that ends with fewer than 60,000 unique rows is dropped (reported, not replaced).
Second confirmatory round (2026-11-19): per-pixel min-max still creates outliers (a pixel almost never lit in training
is stretched to [0, 1], and a test image lighting it lands far from all training data), so the round uses the standard
scaling instead: mnist255 / fashion255 = pixel intensities / 255, no per-feature scaling (names mnist_255, fashion_255).

Raw downloads are untrusted data: only parsed (never executed); run with python -I.
usage: python -I datasets_extra.py prepare NAME --raw RAW_DIR --out DATA_ROOT     (NAME in PREP)
"""
import argparse, bz2, gzip, io, os, sys, zipfile

import numpy as np

sys.path.insert(0, os.environ.get("EXPERIMENTS_DIR",                               # datasets.py (trusted code)
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import datasets as D  # noqa: E402

MIN_UNIQUE = 60_000

URLS = {
    "mnist": ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"],
    "fashion": ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"],
    "corel": ["https://archive.ics.uci.edu/static/public/119/corel+image+features.zip"],
    "kegg": ["https://archive.ics.uci.edu/static/public/221/kegg+metabolic+reaction+network+undirected.zip"],
    "road3d": ["https://archive.ics.uci.edu/static/public/246/3d+road+network+north+jutland+denmark.zip"],
    "codrna": ["https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",
               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r"],
    "ijcnn1": ["https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",
               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2"],
    "sensit": ["https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2",
               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2"],
    "higgs": ["https://openml.org/data/v1/download/2063675/higgs.arff"],
}


def _check_unique(X, name):
    u = len(np.unique(X, axis=0))
    if u < MIN_UNIQUE:
        raise SystemExit(f"{name}: {u} unique rows < {MIN_UNIQUE}: dropped by the selection rule")


def _libsvm(paths, d):
    """Dense matrix from LIBSVM-format files (label idx:value ...; 1-based indices; absent = 0). Labels dropped."""
    rows = []
    for p in paths:
        op = bz2.open if p.endswith(".bz2") else open
        with op(p, "rt") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                r = np.zeros(d)
                for t in parts[1:]:
                    i, v = t.split(":")
                    r[int(i) - 1] = float(v)
                rows.append(r)
    return np.asarray(rows, dtype=np.float64)


def _idx_images(path):
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    magic, n, h, w = np.frombuffer(raw[:16], dtype=">i4")
    assert magic == 2051, magic
    return np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, h * w).astype(np.float64)


DIV255 = {"mnist255": "mnist", "fashion255": "fashion"}      # same raw files, intensities / 255


def prep(name, raw, out):
    base = DIV255.get(name, name)
    f = lambda u: os.path.join(raw, os.path.basename(u))
    src = [f(u) for u in URLS[base]]
    meta = dict(source=URLS[base], raw_sha256=[D._sha256(p) for p in src])
    if base == "mnist":
        with np.load(src[0], allow_pickle=False) as z:
            X = np.vstack([z["x_train"].reshape(-1, 784), z["x_test"].reshape(-1, 784)]).astype(np.float64)
        meta.update(license="CC BY-SA 3.0 (LeCun, Cortes, Burges)", note="train + test images, 784 pixels")
        out_name, scaling = "mnist_mm", "mm"
    elif base == "fashion":
        X = np.vstack([_idx_images(p) for p in src])
        meta.update(license="MIT (Zalando Research)", note="train + test images, 784 pixels")
        out_name, scaling = "fashion_mm", "mm"
    elif name == "corel":
        with zipfile.ZipFile(src[0]) as z:
            names = [n for n in z.namelist() if n.endswith("CoocTexture.asc.gz")]
            assert len(names) == 1, z.namelist()
            text = gzip.decompress(z.open(names[0]).read()).decode("ascii")
        A = np.array([[float(v) for v in r.split()] for r in text.strip().split("\n")])
        X = A[:, 1:]                                   # first column: image id
        assert X.shape[1] == 16, X.shape
        meta.update(license="CC BY 4.0 (UCI 119)", note="CoocTexture.asc, 16 texture features (image id dropped)")
        out_name, scaling = "corel_z", "z"
    elif name == "kegg":
        with zipfile.ZipFile(src[0]) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".data") or n.lower().endswith(".txt")]
            assert len(names) == 1, z.namelist()
            text = z.open(names[0]).read().decode("latin-1")
        rows = []
        for r in text.strip().split("\n"):
            p = r.strip().split(",")
            if "?" in p[1:]:
                continue                               # rows with missing values are dropped
            rows.append([float(v) for v in p[1:]])     # first column: pathway name
        X = np.asarray(rows)
        meta.update(license="CC BY 4.0 (UCI 221)", note="all numeric columns; pathway name dropped; rows with ? dropped")
        out_name, scaling = "kegg_z", "z"
    elif name == "road3d":
        with zipfile.ZipFile(src[0]) as z:
            names = [n for n in z.namelist() if n.endswith(".txt") or n.endswith(".csv")]
            assert len(names) == 1, z.namelist()
            text = z.open(names[0]).read().decode("ascii")
        A = np.array([[float(v) for v in r.split(",")] for r in text.strip().split("\n") if r[0].isdigit()])
        X = A[:, 1:4]                                  # longitude, latitude, altitude (OSM id dropped)
        meta.update(license="CC BY 4.0 (UCI 246)", note="longitude, latitude, altitude")
        out_name, scaling = "road3d_z", "z"
    elif name == "codrna":
        X = _libsvm(src, 8)
        meta.update(license="LIBSVM data sets (Uzilov et al. 2006)", note="cod-rna + .t + .r, labels dropped")
        out_name, scaling = "codrna_z", "z"
    elif name == "ijcnn1":
        X = _libsvm(src, 22)
        meta.update(license="LIBSVM data sets (Prokhorov 2001)", note="ijcnn1 + ijcnn1.t, labels dropped")
        out_name, scaling = "ijcnn1_z", "z"
    elif name == "sensit":
        X = _libsvm(src, 100)
        meta.update(license="LIBSVM data sets (Duarte and Hu 2004)", note="SensIT Vehicle combined + .t, labels dropped")
        out_name, scaling = "sensit_z", "z"
    elif name == "higgs":
        rows, data = [], False
        with open(src[0], "rt") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("%"):
                    continue
                if not data:
                    data = s.lower().startswith("@data")
                    continue
                p = s.split(",")
                if "?" in p:
                    continue                           # rows with missing values are dropped
                rows.append([float(v) for v in p[1:]])  # first attribute: the class
        X = np.asarray(rows)
        assert X.shape[1] == 28, X.shape
        meta.update(license="CC BY 4.0 (UCI HIGGS; OpenML 23512 subsample)", note="28 features, class dropped")
        out_name, scaling = "higgs_z", "z"
    else:
        raise KeyError(name)
    if name in DIV255:
        X = X / 255.0
        out_name, scaling = out_name.replace("_mm", "_255"), "none"
        meta.update(note=meta["note"] + "; intensities / 255, no per-feature scaling")
    X = X[np.isfinite(X).all(1)]
    _check_unique(X, name)
    D._finish(X, out, out_name, scaling, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "urls"])
    ap.add_argument("name")
    ap.add_argument("--raw", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "urls":
        print("\n".join(URLS[a.name]))
    else:
        prep(a.name, a.raw, a.out)


if __name__ == "__main__":
    main()
