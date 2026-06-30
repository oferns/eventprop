"""Configuration, paths, and backend imports for EventProp Visualizer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "src" / "EventProp.Python"))
sys.path.insert(0, str(ROOT / "src" / "EventProp.Native.Python"))

# Pybind11 (Eigen) backend
from eventprop.eventprop_cpp import Spikes as PySpikes, SpikesVector as PySpikesVector
from eventprop.lif_layer import LIFLayerParameters as PyLIFParams, LIFLayer as PyLIFLayer
from eventprop.loss_layer import TTFSCrossEntropyLoss as PyTTFS, TTFSCrossEntropyLossParameters as PyTTFSParams

# Cython (C) backend
from eventprop_c.spikes import Spikes as CySpikes, SpikesVector as CySpikesVector
from eventprop_c.lif_layer import LIFLayerParameters as CyLIFParams, LIFLayer as CyLIFLayer
from eventprop_c.loss_layer import TTFSCrossEntropyLoss as CyTTFS, TTFSCrossEntropyLossParameters as CyTTFSParams

# YinYang TTFS architecture
CFG = dict(
    data_dir=ROOT / "src" / "EventProp.Python" / "eventprop" / "yin_yang_data_set" / "publication_data",
    ckpt_dir=ROOT / "src" / "EventProp.Python" / "paper",
    n_in=5, n_hidden=200, n_out=3, t_max_ms=30.0, n_seeds=10,
)

import pickle as _pkl

def load_all_checkpoints():
    """Load all checkpoint seeds. Returns list of (weights_h, weights_o) per epoch.

    Each entry: ((wh_epoch0, wo_epoch0), (wh_epoch1, wo_epoch1), ...)
    wh shape=(5,200), wo shape=(200,3).
    """
    ckpts = []
    for i in range(CFG["n_seeds"]):
        path = CFG["ckpt_dir"] / f"yinyang_{i}.pkl"
        with open(path, "rb") as f:
            data = _pkl.load(f)
        # data[8] is list of (hidden_weights, output_weights) tuples per epoch
        ckpts.append(data[8])
    return ckpts

LIF = dict(tau_mem=20e-3, tau_syn=5e-3, v_th=1.0)

TRAIN = dict(
    lr=5e-3, epochs=50, minibatch=32,
    alpha=0.003, tau0=0.0005, tau1=0.0064,
    lr_decay=0.95, h_mean=1.5, h_std=0.78, o_mean=0.93, o_std=0.1,
)
