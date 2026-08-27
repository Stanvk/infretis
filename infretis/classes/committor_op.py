"""Online committor order parameter for infRETIS.

The committor ``q(x)`` is the order parameter: ``q = sigmoid(net(phi))`` in the
barrier, and a fixed sentinel value inside each state so infRETIS' interface-
based state detection stays exact.

The class is split in two, deliberately:

* the **estimator** (:class:`MeanValueCommittor`) -- a pure-PyTorch net plus a
  training loop. Swappable: point ``estimator_module`` / ``estimator_class`` at
  any class with the same small interface.
* the **order parameter** (:class:`CommittorOrderParameter`) -- wraps a
  user-supplied *provider* (feature + state definitions) and an estimator; this
  is what infRETIS calls.

The **provider** is one user class (``base_module`` / ``base_class``) with:

    calculate(system) -> phi       # feature vector (already scaled)
    in_A(phi) -> bool              # is phi in state A
    in_B(phi) -> bool              # is phi in state B
    sample_states(n) -> (a, b)     # optional; n feature vectors in A and in B
                                   # (only needed when pretrain=true)

Features and states are the user's responsibility; this class only turns them
into a committor. Example toml::

    [orderparameter]
    class = "CommittorOrderParameter"
    module = "committor_op.py"
    base_module = "wqop.py"        # the provider (features + states)
    base_class = "wqop"
    checkpoint = "committor_net.pt"
    pretrain = true                # pretrain the BCs from sample_states

Interfaces must be committor values with the states outside ``(0, 1)`` -- e.g.
``interfaces = [-0.5, ..., 1.5]`` -- so sentinels ``q=-1`` (A) / ``q=2`` (B)
fall below / above the bounding interfaces.
"""
from __future__ import annotations

import json
import os
import time
from typing import List

import numpy as np
import torch
from torch import nn

from infretis.classes.system import System
from infretis.classes.orderparameter import OrderParameter
from infretis.core.core import import_from

# Sentinels returned inside the states: outside the barrier range (0, 1) and
# outside the bounding interfaces [-0.5, 1.5], so state detection is exact.
Q_IN_A = -1.0
Q_IN_B = 2.0


class CommittorNet(nn.Module):
    """MLP mapping features -> committor log-odds z (q = sigmoid(z))."""

    def __init__(self, input_dim: int, hidden_dims=(64, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MeanValueCommittor:
    """Mean-value committor estimator (pure PyTorch).

    A sample is one two-way shot: the shooting-point features ``phi_sp``, the
    two endpoint features ``phi_b`` / ``phi_f`` and their labels (0.0 in A, 1.0
    in B, NaN on a fence -> bootstrap from the model). The loss regresses the
    shooting-point committor onto the mean of the two endpoint committors. The
    net is built lazily on the first sample/prediction.

    Interface the order parameter relies on: ``predict``, ``observe``,
    ``update``, ``clear_buffer``, ``save``/``load``, ``reset``,
    ``n_samples``.
    """

    def __init__(self, hidden_dims=(64, 64), lr=1e-3, weight_decay=1e-4,
                 batch_size=128, seed=0, device="cpu") -> None:
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.net = None
        self.opt = None
        self._buf = []   # list of (phi_sp, phi_b, phi_f, lab_b, lab_f)

    def _ensure_net(self, dim: int) -> None:
        if self.net is None:
            torch.manual_seed(self.seed)
            self.net = CommittorNet(dim, self.hidden_dims).to(self.device)
            self.opt = torch.optim.AdamW(
                self.net.parameters(), lr=self.lr,
                weight_decay=self.weight_decay,
            )

    def reset(self) -> None:
        """Empty net, optimizer and buffer (a fresh start before pretrain)."""
        self.net = self.opt = None
        self._buf = []
        self.rng = np.random.default_rng(self.seed)

    def clear_buffer(self) -> None:
        """Empty the sample buffer but keep the (warm-started) net/optimizer."""
        self._buf = []

    @property
    def n_samples(self) -> int:
        return len(self._buf)

    def predict(self, phi) -> np.ndarray:
        """Committor q = sigmoid(net(phi)); accepts (D,) or (N, D)."""
        x = np.atleast_2d(np.asarray(phi, dtype=np.float64))
        self._ensure_net(x.shape[1])
        self.net.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            z = self.net(xt).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def observe(self, phi_sp, phi_b, phi_f, lab_b, lab_f) -> None:
        """Append one two-way-shot sample."""
        s = (np.asarray(phi_sp, dtype=np.float64),
             np.asarray(phi_b, dtype=np.float64),
             np.asarray(phi_f, dtype=np.float64), float(lab_b), float(lab_f))
        self._ensure_net(s[0].shape[0])
        self._buf.append(s)

    def update(self, steps=1):
        """Run ``steps`` gradient steps of the mean-value loss."""
        if self.net is None or self.n_samples < 2:
            return None
        self.net.train()
        n = self.n_samples
        bs = min(self.batch_size, n)
        loss = None
        for _ in range(int(steps)):
            idx = self.rng.choice(n, size=bs, replace=False)
            loss = self._mv_loss(idx)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return None if loss is None else float(loss.detach())

    def _endpoint_q(self, feats, labs) -> torch.Tensor:
        # hard label where known; detached bootstrap where NaN (a fence end)
        t = torch.as_tensor(labs, dtype=torch.float32, device=self.device)
        nan = torch.isnan(t)
        if bool(nan.any()):
            x = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                boot = torch.sigmoid(self.net(x))
            t = torch.where(nan, boot, t)
        return t

    def _mv_loss(self, idx) -> torch.Tensor:
        sp = np.stack([self._buf[i][0] for i in idx])
        eb = np.stack([self._buf[i][1] for i in idx])
        ef = np.stack([self._buf[i][2] for i in idx])
        lb = np.array([self._buf[i][3] for i in idx])
        lf = np.array([self._buf[i][4] for i in idx])
        z = self.net(
            torch.as_tensor(sp, dtype=torch.float32, device=self.device)
        )
        target = 0.5 * (self._endpoint_q(eb, lb) + self._endpoint_q(ef, lf))
        sp_pos = torch.nn.functional.softplus(z)     # -log(1 - q)
        sp_neg = torch.nn.functional.softplus(-z)    # -log q
        return (target * sp_neg + (1.0 - target) * sp_pos).mean()

    def save(self, path: str) -> None:
        blob = {
            "input_dim": None if self.net is None else self.net.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "net": None if self.net is None else self.net.state_dict(),
            "opt": None if self.opt is None else self.opt.state_dict(),
            "buffer": self._buf,
            "rng": self.rng.bit_generator.state,
        }
        tmp = f"{path}.tmp.{os.getpid()}"    # atomic write: a reader (the
        torch.save(blob, tmp)                # propagator) never sees a partial
        os.replace(tmp, path)                # checkpoint

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self._buf = list(blob.get("buffer", []))
        if blob.get("rng") is not None:
            self.rng.bit_generator.state = blob["rng"]
        if blob.get("input_dim") is not None:
            self.hidden_dims = tuple(blob["hidden_dims"])
            self._ensure_net(int(blob["input_dim"]))
            self.net.load_state_dict(blob["net"])
            if blob.get("opt") is not None:
                self.opt.load_state_dict(blob["opt"])


class CommittorOrderParameter(OrderParameter):
    """Committor order parameter: q via an estimator, states via a provider.

    ``calculate`` returns ``[q, *phi]``: ``Q_IN_A`` inside A, ``Q_IN_B`` inside
    B (fixed sentinels), else the estimator's committor. The provider
    (``base_module`` / ``base_class``) supplies the features and the state
    tests; the estimator (``estimator_module`` / ``estimator_class``, default
    :class:`MeanValueCommittor`) supplies the network.
    """

    def __init__(self, base_module, base_class, base_kwargs=None,
                 estimator_module="", estimator_class="",
                 checkpoint="committor_net.pt", pretrain=False,
                 pretrain_n=2000, pretrain_steps=4000,
                 label_smoothing=0.02, anchor_ratio=4.0,
                 hidden_dims=(64, 64), train_steps=2000, batch_size=128,
                 lr=1e-3, weight_decay=1e-4, seed=0, device="cpu") -> None:
        super().__init__(
            description="Online committor (NN) order parameter",
            velocity=False,
        )
        # provider: the user's features + state definitions
        self.provider = import_from(base_module, base_class)(
            **(base_kwargs or {})
        )
        # estimator: the network (default = mean-value, defined above)
        est_cls = (
            import_from(estimator_module, estimator_class)
            if estimator_module and estimator_class
            else MeanValueCommittor
        )
        self.estimator = est_cls(
            hidden_dims=hidden_dims, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, seed=seed, device=device,
        )
        self.checkpoint = checkpoint
        self.train_steps = int(train_steps)
        self.pretrain_n = int(pretrain_n)
        self.pretrain_steps = int(pretrain_steps)
        # label smoothing: soft BC targets bound the logits so the committor
        # cannot collapse into a saturated step; sentinels still give exact
        # state detection, so q_lo/q_hi at deep A/B are irrelevant to TIS.
        self.q_lo = float(label_smoothing)
        self.q_hi = 1.0 - float(label_smoothing)
        # anchor_ratio caps the re-added BC anchors to ~anchor_ratio*n_shots per
        # state during train_from_shots, so the shots are not drowned by BCs.
        self.anchor_ratio = float(anchor_ratio)
        # restart from a checkpoint, else optionally pretrain the BCs once
        if checkpoint and os.path.isfile(checkpoint):
            self.estimator.load(checkpoint)
        elif pretrain and checkpoint:
            self._pretrain_guarded()

    def _pretrain_guarded(self) -> None:
        """Pretrain the BCs once, safe across processes (propagator+workers).

        The first process to grab the lock resets the net, samples the states
        (``provider.sample_states``), pins q=0/1 on them and trains; the others
        wait for the checkpoint and load it.
        """
        lock = self.checkpoint + ".lock"
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            for _ in range(1200):
                if os.path.isfile(self.checkpoint):
                    return self.estimator.load(self.checkpoint)
                time.sleep(0.1)
            raise RuntimeError(f"timed out waiting for '{self.checkpoint}'")
        try:
            self.estimator.reset()
            feats_a, feats_b = self.provider.sample_states(self.pretrain_n)
            self._add_anchors(feats_a, feats_b)
            self.estimator.update(self.pretrain_steps)
            self.estimator.save(self.checkpoint)
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass

    def reload_weights(self) -> bool:
        """Reload the net from the checkpoint (ase_external propagator sync)."""
        if self.checkpoint and os.path.isfile(self.checkpoint):
            self.estimator.load(self.checkpoint)
            return True
        return False

    def _label(self, phi) -> float:
        """q_lo in A, q_hi in B, NaN in the barrier (a fence end -> bootstrap).

        Label-smoothed (q_lo/q_hi, not 0/1) so committed ends, like the BC
        anchors, cannot drive the logits to saturation.
        """
        if self.provider.in_A(phi):
            return self.q_lo
        if self.provider.in_B(phi):
            return self.q_hi
        return float("nan")

    def _add_anchors(self, feats_a, feats_b) -> None:
        """Add persistent BC anchors with smoothed targets (q_lo/q_hi)."""
        for p in np.atleast_2d(np.asarray(feats_a, dtype=np.float64)):
            self.estimator.observe(p, p, p, self.q_lo, self.q_lo)
        for p in np.atleast_2d(np.asarray(feats_b, dtype=np.float64)):
            self.estimator.observe(p, p, p, self.q_hi, self.q_hi)

    def calculate(self, system: System) -> List[float]:
        phi = np.asarray(self.provider.calculate(system), dtype=np.float64)
        if self.provider.in_A(phi):
            q = Q_IN_A
        elif self.provider.in_B(phi):
            q = Q_IN_B
        else:
            q = float(self.estimator.predict(phi)[0])
        return [q, *phi.tolist()]

    def recompute_order(self, orders):
        """Recompute q of stored ``[q, *phi]`` rows with the current net.

        Feature-cached: ``phi`` is a function of the configuration only, so no
        trajectory read is needed -- only the net-dependent q changes. States
        keep their sentinels; barrier frames get the current net's q. Called by
        infinit's interface refresh (via ``load_path``) so interface placement
        rests on a single, current committor.
        """
        rows = np.atleast_2d(np.asarray(orders, dtype=np.float64)).copy()
        phi = rows[:, 1:]
        q = self.estimator.predict(phi)
        for i in range(rows.shape[0]):
            if self.provider.in_A(phi[i]):
                rows[i, 0] = Q_IN_A
            elif self.provider.in_B(phi[i]):
                rows[i, 0] = Q_IN_B
            else:
                rows[i, 0] = q[i]
        return rows

    def _observe_record(self, sp_order, bwd_order, fwd_order, fwd_ok):
        """Add one two-way shot to the estimator buffer (no training here).

        Order vectors are ``[q, *phi]``; endpoints are labelled by the
        provider's state tests on ``phi`` (0 in A, 1 in B, NaN on a fence ->
        bootstrap). The stored q is ignored -- features and labels are
        config-only, so old records train correctly against the current net.
        """
        phi_sp = np.asarray(sp_order[1:], dtype=np.float64)
        phi_b = np.asarray(bwd_order[1:], dtype=np.float64)
        phi_f = np.asarray(fwd_order[1:], dtype=np.float64)
        lab_b = self._label(phi_b)
        lab_f = self._label(phi_f) if fwd_ok else float("nan")
        self.estimator.observe(phi_sp, phi_b, phi_f, lab_b, lab_f)

    def train_from_shots(self, shots_file="shots.jsonl", steps=None):
        """Retrain the net from the aggregated shot log (per iteration).

        Called once at the infinit boundary, in a single process. Rebuilds the
        buffer from scratch -- the persistent BC anchors (``sample_states``)
        plus every shot in ``shots_file`` -- warm-starting from the current
        net, then runs ``steps`` gradient steps and saves the checkpoint. This
        replaces per-shot worker training: one trainer, one checkpoint writer,
        so it is multi-worker safe and gives a fixed OP within each iteration.
        """
        if steps is None:
            steps = self.train_steps
        self.estimator.clear_buffer()
        # replay every recorded shot (per-jump fence ends + committed ends)
        n_shots = 0
        if os.path.isfile(shots_file):
            with open(shots_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._observe_record(
                            rec["sp_order"], rec["bwd_end"],
                            rec["fwd_end"], rec["fwd_ok"],
                        )
                        n_shots += 1
                    except (ValueError, KeyError, TypeError):
                        continue
        # re-pin the BCs, but cap the anchors so they do not drown the shots:
        # ~anchor_ratio*n_shots per state (floored so the BCs stay pinned early,
        # capped at pretrain_n).
        n_anch = int(min(self.pretrain_n,
                         max(50, self.anchor_ratio * max(n_shots, 1))))
        feats_a, feats_b = self.provider.sample_states(n_anch)
        self._add_anchors(feats_a, feats_b)
        loss = self.estimator.update(steps)
        if self.checkpoint:
            self.estimator.save(self.checkpoint)
        return {"n_shots": n_shots, "n_anchors": 2 * n_anch,
                "n_samples": self.estimator.n_samples, "loss": loss}
