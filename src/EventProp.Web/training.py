"""Training loop logic for both backends."""
import numpy as np
from config import (CFG, LIF, TRAIN, CySpikes, CySpikesVector, CyLIFParams, CyLIFLayer, CyTTFS, CyTTFSParams,
                    PySpikes, PySpikesVector, PyLIFParams, PyLIFLayer, PyTTFS, PyTTFSParams)


def _enc_cy(samp):
    sl = []
    for x in samp:
        t = np.array([v*30e-3 for v in list(x)+[0.0]], dtype=np.float64)
        s = np.arange(len(x)+1, dtype=np.int32); i = np.argsort(t)
        sl.append(CySpikes(t[i], s[i]))
    return CySpikesVector(sl)


def _enc_py(samp):
    sl = []
    for x in samp:
        t = np.array([v*30e-3 for v in list(x)+[0.0]], dtype=np.float64).reshape(-1, 1)
        s = np.arange(len(x)+1, dtype=np.int32).reshape(-1, 1); i = np.argsort(t.ravel())
        sl.append(PySpikes(t[i], s[i]))
    return PySpikesVector(sl)


def run_training(seed, n_epochs):
    """Run full training on both backends from same seed.

    Returns: cy_accs, py_accs, cy_losses, py_losses, wh_cy, wo_cy, wh_py, wo_py
    """
    np.random.seed(seed)
    wh_cy = np.random.normal(TRAIN["h_mean"], TRAIN["h_std"], (CFG["n_in"], CFG["n_hidden"]))
    wo_cy = np.random.normal(TRAIN["o_mean"], TRAIN["o_std"], (CFG["n_hidden"], CFG["n_out"]))
    wh_py, wo_py = wh_cy.copy(), wo_cy.copy()

    tr_s = np.load(CFG["data_dir"] / "train_samples.npy")
    tr_l = np.load(CFG["data_dir"] / "train_labels.npy")
    n_train = len(tr_s)

    cy_accs, py_accs, cy_losses, py_losses = [], [], [], []
    lr = TRAIN["lr"]

    for epoch in range(n_epochs):
        idx = np.random.permutation(n_train)
        cy_ea, py_ea, cy_el, py_el = 0., 0., 0., 0.
        n_batches = 0
        for start in range(0, n_train, TRAIN["minibatch"]):
            end = min(start + TRAIN["minibatch"], n_train)
            batch_idx = idx[start:end]
            bs = end - start
            batch_s, batch_l = tr_s[batch_idx], tr_l[batch_idx]

            inp_cy, inp_py = _enc_cy(batch_s), _enc_py(batch_s)

            lp_cy = CyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
            lo_cy = CyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
            lp_py = PyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
            lo_py = PyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)

            hl_cy = CyLIFLayer(lp_cy, w_in=wh_cy)
            ol_cy = CyTTFS(CyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_cy), w_in=wo_cy)
            hl_py = PyLIFLayer(lp_py, w_in=wh_py)
            ol_py = PyTTFS(PyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_py), w_in=wo_py)

            ho_cy = hl_cy.forward(inp_cy); ol_cy.forward(ho_cy)
            cy_ea += ol_cy.get_accuracy(batch_l)*bs; cy_el += np.mean(ol_cy.get_losses(batch_l))*bs
            ol_cy.backward(batch_l); hl_cy.backward()
            wh_cy -= lr*hl_cy.gradient; wo_cy -= lr*ol_cy.gradient

            ho_py = hl_py.forward(inp_py); ol_py.forward(ho_py)
            py_ea += ol_py.get_accuracy(batch_l)*bs; py_el += np.mean(ol_py.get_losses(batch_l))*bs
            ol_py.backward(batch_l); hl_py.backward()
            wh_py -= lr*hl_py.gradient; wo_py -= lr*ol_py.gradient

            n_batches += 1

        cy_accs.append(cy_ea/n_train); py_accs.append(py_ea/n_train)
        cy_losses.append(cy_el/n_train); py_losses.append(py_el/n_train)
        lr *= TRAIN["lr_decay"]

    return np.array(cy_accs), np.array(py_accs), np.array(cy_losses), np.array(py_losses), wh_cy, wo_cy, wh_py, wo_py

