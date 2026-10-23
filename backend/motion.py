"""
Optical-flow estimators  (Notebook 02_Motion_Model  ->  models/flow_selection.json (+ models/motion_model.pth if MotionNetV2 was selected))

Verbatim copies of: work_size, ZeroFlow, Farneback, DIS, FlowRefineNet (build_refine_net), MotionNetEstimator, RAFTSmall.
The estimator that is USED is whatever notebook 02 selected on validation (flow_selection.json -> "selected").  For your trained run: DIS @ work width 640.
"""
import os
import numpy as np
import cv2

from common import pick_device


# --- notebook 04, cell 10: work_size --------------------
def work_size(orig_w, orig_h, work_width=320):
    w = int(work_width); h = int(round(orig_h * w / orig_w / 2.0) * 2); return w, max(h, 16)


# --- notebook 04, cell 13: ZeroFlow --------------------
class ZeroFlow:
    name = "zero"
    def __call__(self, a, b): return np.zeros(a.shape[:2] + (2,), np.float32)


# --- notebook 04, cell 13: Farneback --------------------
class Farneback:
    name = "farneback"
    def __init__(self, levels=4, winsize=21, iters=4, poly_n=7, poly_sigma=1.5):
        self.p = dict(pyr_scale=0.5, levels=levels, winsize=winsize, iterations=iters, poly_n=poly_n, poly_sigma=poly_sigma, flags=0)
    def __call__(self, a, b):
        g1, g2 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(g1, g2, None, **self.p).astype(np.float32)


# --- notebook 04, cell 13: DIS --------------------
class DIS:
    name = "dis"
    def __init__(self, preset="medium", finest_scale=0):
        pr = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[preset]
        self.d = cv2.DISOpticalFlow_create(pr)
        try: self.d.setFinestScale(int(finest_scale)); self.d.setUseSpatialPropagation(True)
        except Exception: pass
    def __call__(self, a, b):
        g1, g2 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        return self.d.calc(g1, g2, None).astype(np.float32)


# --- notebook 04, cell 13: classical_estimators --------------------
def classical_estimators(): return {"zero": ZeroFlow(), "farneback": Farneback(), "dis": DIS()}


# --- notebook 04, cell 13: _torch --------------------
def _torch():
    import torch, torch.nn as nn, torch.nn.functional as F
    return torch, nn, F


# --- notebook 04, cell 13: build_refine_net --------------------
def build_refine_net(use_prior=True, base=24):
    torch, nn, F = _torch()

    def block(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class FlowRefineNet(nn.Module):
        """Two frames (+ optional classical prior flow) -> flow = prior + learned residual.  Multi-scale U-Net, zero-initialised head
        (so training starts exactly at the classical prior and can only improve on it)."""
        def __init__(self):
            super().__init__(); self.use_prior = use_prior; cin = 9 + (2 if use_prior else 0)
            self.e1, self.e2, self.e3, self.e4 = block(cin, base), block(base, base * 2, 2), block(base * 2, base * 4, 2), block(base * 4, base * 6, 2)
            self.d3, self.d2, self.d1 = block(base * 6 + base * 4, base * 4), block(base * 4 + base * 2, base * 2), block(base * 2 + base, base)
            self.head = nn.Conv2d(base, 2, 3, 1, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        def forward(self, x1, x2, prior=None):
            H, W = x1.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
            n1, n2 = (x1 - 0.5) / 0.25, (x2 - 0.5) / 0.25; parts = [n1, n2, (n1 - n2).abs()]
            if self.use_prior: parts.append(prior * 4.0)
            x = torch.cat(parts, 1)
            if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
            up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            d3 = self.d3(torch.cat([up(e4, e3), e3], 1)); d2 = self.d2(torch.cat([up(d3, e2), e2], 1)); d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
            res = self.head(d1)[..., :H, :W]
            return (prior + res) if self.use_prior else res

    return FlowRefineNet()


# --- notebook 04, cell 13: MotionNetEstimator --------------------
class MotionNetEstimator:
    """Callable wrapper around a trained FlowRefineNet checkpoint (written by train_motion_net)."""
    name = "motionnet_v2"
    def __init__(self, ckpt_path, device=None):
        torch, _, _ = _torch(); self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False); self.meta = ck.get("meta", {})
        self.use_prior = bool(ck["use_prior"]); self.net = build_refine_net(self.use_prior, ck.get("base", 24)); self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.net.to(self.dev); self.prior = DIS() if self.use_prior else None
        self.work_width = ck.get("work_width")
    def __call__(self, a, b):
        torch = self.torch
        with torch.no_grad():
            t = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float().div(255.0).to(self.dev)
            pr = None
            if self.use_prior: pr = torch.from_numpy(self.prior(a, b)).permute(2, 0, 1)[None].float().to(self.dev)
            out = self.net(t(a), t(b), pr)
        return out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


# --- notebook 04, cell 13: RAFTSmall --------------------
class RAFTSmall:
    """Optional pretrained RAFT-small (torchvision). Downloads weights on first use; returns None from `create` if unavailable."""
    name = "raft_small"
    @staticmethod
    def create(device=None):
        try:
            torch, _, F = _torch(); from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self = RAFTSmall(); self.torch, self.F = torch, F; self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).eval().to(self.dev); return self
        except Exception as e:
            print(f"[info] RAFT-small unavailable ({type(e).__name__}: {str(e)[:80]}). Skipping it."); return None
    def __call__(self, a, b):
        torch, F = self.torch, self.F; H, W = a.shape[:2]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        with torch.no_grad():
            t = lambda im: F.pad(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float().div(255.0).mul(2).sub(1), (0, pw, 0, ph), mode="replicate").to(self.dev)
            out = self.model(t(a), t(b))[-1]
        return out[0, :, :H, :W].permute(1, 2, 0).cpu().numpy().astype(np.float32)


# ============================================================================================================================
# BACKEND-ONLY
# ============================================================================================================================
def load_selected_estimator(name, device, motion_model_path):
    """Same logic as notebook 04 cell 33 `load_selected_estimator`; the only change is that the checkpoint path is an argument instead of a notebook global."""
    if name == "motionnet_v2": return MotionNetEstimator(motion_model_path, device)
    if name == "raft_small":
        est = RAFTSmall.create(device)
        if est is not None: return est
        print("[warn] RAFT-small unavailable -> DIS"); return DIS()
    return classical_estimators().get(name, DIS())
