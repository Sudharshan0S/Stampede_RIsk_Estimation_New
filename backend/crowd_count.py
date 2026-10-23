"""
Crowd-count model  (Notebook 01_Crowd_Count_Model  ->  models/crowd_count_model.pth)

EVERY definition below is copied VERBATIM from the notebooks (01 cells 11/12/22/23, identical copies in 03/04):
    resizing (min_long / max_long / tiling), exposure normalisation, ImageNet mean/std normalisation, the CrowdPointNet layers,
    anchors, point NMS, accept/uncertain split, regional refinement and the CountModel wrapper.
Do not edit the logic here - change the notebook and re-export instead, otherwise training and inference diverge.
"""
import os, math, copy, hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter

from common import pick_device, ceil_to


# --- notebook 04, cell 6: DEFAULT_REFINE --------------------
DEFAULT_REFINE = dict(
    enabled=True, region_size=128,      # px (original image) of the analysis grid cells
    min_region_density=6.0,             # ignore regions whose density estimate is below this many people
    abs_gap=3.0, rel_gap=0.30,          # flag if (density - accepted points) >= max(abs_gap, rel_gap*density)
    min_uncertain=3,                    # ... or if >= this many uncertain (yellow) points sit in the region
    crop_margin=32,                     # px of context added around the region (overlap with neighbours)
    scale=2.0, scale_growth=1.5, max_scale=3.0,   # effective-resolution multiplier on iteration 0, growth per iteration, hard cap
    max_iters=2, max_regions=12,        # max refinement iterations / max regions re-analysed per image
    max_crop_pixels=1.5e6,              # memory guard for the up-scaled crop
    policy="replace_if_more")           # 'union' | 'replace_if_more' | 'replace'  (see _merge_region)


# --- notebook 04, cell 6: DEFAULT_INFER --------------------
DEFAULT_INFER = dict(
    density_scale=10.0, min_long=640, max_long=1536, max_pixels=2.0e6, tile=768, tile_overlap=128,
    exposure_norm=True, exposure_target=0.42, min_candidate_conf=0.05, peak_min_value=0.01, amp=False,
    conf_accept=0.5, uncertain_ratio=0.5, dup_radius=3.0, refine=copy.deepcopy(DEFAULT_REFINE))


# --- notebook 04, cell 21: IMNET_MEAN --------------------
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32); IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --- notebook 04, cell 21: _f --------------------
def _f(img): return img.astype(np.float32) / 255.0


# --- notebook 04, cell 21: _u8 --------------------
def _u8(x): return np.ascontiguousarray(np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8))


# --- notebook 04, cell 21: normalize_exposure --------------------
def normalize_exposure(img, target=0.42):
    """Gamma + percentile stretch so dark / bright photos reach the network in a similar range.
    Applied in training (with a small skip probability) AND in every inference path."""
    x = _f(img); lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]; m = float(lum[::2, ::2].mean())
    if m < 0.30 or m > 0.62:
        g = float(np.clip(np.log(target) / np.log(np.clip(m, 1e-3, 0.999)), 0.35, 2.5))
        x = np.power(x, g); lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    lo, hi = np.percentile(lum[::2, ::2], (0.5, 99.5))
    if hi - lo > 0.08: x = 0.5 * x + 0.5 * np.clip((x - lo) / (hi - lo), 0, 1)
    return _u8(x)


# --- notebook 04, cell 21: to_tensor --------------------
def to_tensor(img): return torch.from_numpy(((_f(img) - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())


# --- notebook 04, cell 23: conv_bn --------------------
def conv_bn(i, o, k=3, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


# --- notebook 04, cell 23: make_anchors --------------------
def make_anchors(h, w, stride, spacing, device="cpu"):
    """Anchor (x, y) pixel coordinates, ordering (row i, col j, sub-anchor k) -> (h*w*K, 2).
    Each feature cell holds K=(stride/spacing)^2 anchors on a regular grid with `spacing` pixels between them."""
    na = stride // spacing
    sub = (torch.arange(na, dtype=torch.float32, device=device) + 0.5) * spacing
    sy, sx = torch.meshgrid(sub, sub, indexing="ij"); oy, ox = sy.reshape(-1), sx.reshape(-1); K = na * na
    cy = torch.arange(h, dtype=torch.float32, device=device) * stride; cx = torch.arange(w, dtype=torch.float32, device=device) * stride
    Y = (cy[:, None, None] + oy[None, None, :]).expand(h, w, K); X = (cx[None, :, None] + ox[None, None, :]).expand(h, w, K)
    return torch.stack([X, Y], -1).reshape(-1, 2)


# --- notebook 04, cell 23: MODEL_SPEC_KEYS --------------------
MODEL_SPEC_KEYS = ("point", "density", "point_stride", "anchor_spacing", "fuse", "cls_prior")


# --- notebook 04, cell 23: CrowdPointNet --------------------
class CrowdPointNet(nn.Module):
    """forward(x) -> dict:
         'density' : (B,1,H/8,W/8)  non-negative map, sum/density_scale = count          (if density branch on)
         'logits'  : (B,A,2)        [background, head] logits for every anchor           (if point branch on)
         'offsets' : (B,A,2)        (dx,dy) in units of `point_stride` pixels
         'anchors' : (A,2)          anchor pixel coordinates in the network-input frame
       predicted head centre = anchor + offset * point_stride."""
    def __init__(self, point=True, density=True, point_stride=8, anchor_spacing=4, fuse=128, pretrained=True, freeze_stem=True, cls_prior=0.02):
        super().__init__()
        assert point or density, "at least one branch is required"
        assert point_stride in (4, 8) and point_stride % anchor_spacing == 0, "point_stride must be 4 or 8 and a multiple of anchor_spacing"
        self.point, self.density, self.point_stride, self.anchor_spacing, self.fuse_ch, self.cls_prior = bool(point), bool(density), point_stride, anchor_spacing, fuse, cls_prior
        try: r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception as e:
            print("[warn] ImageNet weights unavailable -> RANDOM init (training will be much worse):", repr(e)[:120]); r = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.p1, self.p2, self.p3, self.p4 = nn.Conv2d(64, fuse, 1), nn.Conv2d(128, fuse, 1), nn.Conv2d(256, fuse, 1), nn.Conv2d(512, fuse, 1)
        # shared multi-scale fusion at stride 8 (layers 2,3,4) + dilated context block (from notebook 12)
        self.fuse8 = nn.Sequential(conv_bn(fuse * 3, fuse), conv_bn(fuse, fuse))
        self.ctx = nn.ModuleList([conv_bn(fuse, fuse // 2, 3, d) for d in (1, 2, 4)]); self.ctx_merge = conv_bn(fuse // 2 * 3, fuse, 1)
        if self.density:
            self.den_head = nn.Sequential(conv_bn(fuse, 64), conv_bn(64, 32), nn.Conv2d(32, 1, 1)); nn.init.constant_(self.den_head[-1].bias, -5.0)
        if self.point:
            if point_stride == 4: self.fuse4 = nn.Sequential(conv_bn(fuse * 2, fuse), conv_bn(fuse, fuse))   # high-resolution path: up(F8) + layer1
            self.K = (point_stride // anchor_spacing) ** 2
            self.cls_tower = nn.Sequential(conv_bn(fuse, fuse), conv_bn(fuse, fuse)); self.reg_tower = nn.Sequential(conv_bn(fuse, fuse), conv_bn(fuse, fuse))
            self.cls_out = nn.Conv2d(fuse, self.K * 2, 3, padding=1); self.reg_out = nn.Conv2d(fuse, self.K * 2, 3, padding=1)
            nn.init.normal_(self.reg_out.weight, std=0.01); nn.init.zeros_(self.reg_out.bias)
            nn.init.normal_(self.cls_out.weight, std=0.01)
            b = torch.zeros(self.K, 2); b[:, 1] = math.log(cls_prior / (1 - cls_prior)); self.cls_out.bias.data = b.reshape(-1)   # start with p(head)=cls_prior
        self.freeze_stem = freeze_stem
        if freeze_stem:
            for m in (self.stem, self.layer1):
                for p in m.parameters(): p.requires_grad = False
    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_stem:
            for m in (self.stem, self.layer1): m.eval()
        return self
    def forward(self, x):
        f1 = self.layer1(self.stem(x)); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3); hw = f2.shape[-2:]
        up = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
        z = self.fuse8(torch.cat([self.p2(f2), up(self.p3(f3), hw), up(self.p4(f4), hw)], 1))
        z = z + self.ctx_merge(torch.cat([c(z) for c in self.ctx], 1)); out = {}
        if self.density: out["density"] = F.softplus(self.den_head(z))
        if self.point:
            zp = self.fuse4(torch.cat([up(z, f1.shape[-2:]), self.p1(f1)], 1)) if self.point_stride == 4 else z
            cls = self.cls_out(self.cls_tower(zp)); reg = self.reg_out(self.reg_tower(zp)); B, _, h, w = cls.shape
            out["logits"] = cls.view(B, self.K, 2, h, w).permute(0, 3, 4, 1, 2).reshape(B, h * w * self.K, 2)
            out["offsets"] = reg.view(B, self.K, 2, h, w).permute(0, 3, 4, 1, 2).reshape(B, h * w * self.K, 2)
            out["anchors"] = make_anchors(h, w, self.point_stride, self.anchor_spacing, x.device)
        return out


# --- notebook 04, cell 23: build_model --------------------
def build_model(spec, pretrained=True, freeze_stem=True):
    return CrowdPointNet(pretrained=pretrained, freeze_stem=freeze_stem, **{k: spec[k] for k in MODEL_SPEC_KEYS if k in spec})


# --- notebook 04, cell 24: point_nms --------------------
def point_nms(pts, scores, radius):
    """Greedy, confidence-ordered, distance-based duplicate removal.
    A point is removed iff it lies within `radius` pixels of a higher-confidence point that was kept.
    radius<=0 disables it. Complexity O(N log N) with a KD-tree. Returns indices of kept points."""
    n = len(pts)
    if n == 0 or radius <= 0: return np.arange(n)
    order = np.argsort(-scores, kind="stable"); nbrs = cKDTree(pts).query_ball_point(pts, r=radius)
    alive = np.ones(n, bool); keep = []
    for i in order:
        if alive[i]: keep.append(i); alive[nbrs[i]] = False
    return np.asarray(keep, dtype=int)


# --- notebook 04, cell 24: finalize_points --------------------
def finalize_points(pts, scores, conf_accept=0.5, uncertain_ratio=0.5, dup_radius=3.0):
    """Candidate pool -> final structure.  accepted: score>=conf_accept (these ARE the count);
    uncertain: conf_accept*uncertain_ratio <= score < conf_accept (shown, never counted)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2); scores = np.asarray(scores, np.float32).reshape(-1)
    sel = scores >= conf_accept * uncertain_ratio; pts, scores = pts[sel], scores[sel]
    k = point_nms(pts, scores, dup_radius); pts, scores = pts[k], scores[k]; acc = scores >= conf_accept
    return dict(accepted=pts[acc], acc_scores=scores[acc], uncertain=pts[~acc], unc_scores=scores[~acc], count=int(acc.sum()))


# --- notebook 04, cell 24: density_peaks --------------------
def density_peaks(den, cell_xy, min_value=0.01):
    """Local maxima of a density map (counts per cell) -> (pts, scores). Used ONLY as the localisation baseline for the
    density-only model; the point models never use it. Sub-cell position = centroid of the 3x3 neighbourhood."""
    if den is None or den.size == 0: return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    mx = maximum_filter(den, size=3, mode="constant"); ys, xs = np.nonzero((den == mx) & (den >= min_value))
    if len(ys) == 0: return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    pad = np.pad(den, 1); sw = sx = sy = np.zeros(len(ys), np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = pad[ys + 1 + dy, xs + 1 + dx].astype(np.float64); sw = sw + w; sx = sx + w * (xs + dx); sy = sy + w * (ys + dy)
    cx, cy = sx / np.maximum(sw, 1e-12), sy / np.maximum(sw, 1e-12)
    pts = np.stack([(cx + 0.5) * cell_xy[0], (cy + 0.5) * cell_xy[1]], 1).astype(np.float32)
    return pts, np.clip(den[ys, xs], 0, 1).astype(np.float32)


# --- notebook 04, cell 24: region_count --------------------
def region_count(den, cell_xy, box):
    """Sum of density (people) whose cell centre lies in box=(x0,y0,x1,y1) given in ORIGINAL image pixels."""
    if den is None: return float("nan")
    hd, wd = den.shape; cx = (np.arange(wd) + 0.5) * cell_xy[0]; cy = (np.arange(hd) + 0.5) * cell_xy[1]
    j0, j1 = np.searchsorted(cx, box[0], "left"), np.searchsorted(cx, box[2], "left"); i0, i1 = np.searchsorted(cy, box[1], "left"), np.searchsorted(cy, box[3], "left")
    return float(den[i0:i1, j0:j1].sum())


# --- notebook 04, cell 24: inside_box --------------------
def inside_box(pts, box):
    if len(pts) == 0: return np.zeros(0, bool)
    return (pts[:, 0] >= box[0]) & (pts[:, 0] < box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] < box[3])


# --- notebook 04, cell 24: crop_to_original --------------------
def crop_to_original(pts_crop, crop_x0, crop_y0, sx, sy):
    """Crop-local (resized) coordinates -> original-image coordinates.  The crop was cut at (crop_x0,crop_y0)
    and resized by (sx,sy) = (new_w/crop_w, new_h/crop_h), so  x_orig = x_crop / sx + crop_x0."""
    pts_crop = np.asarray(pts_crop, np.float32).reshape(-1, 2)
    return pts_crop / np.array([sx, sy], np.float32) + np.array([crop_x0, crop_y0], np.float32)


# --- notebook 04, cell 24: tile_axis --------------------
def tile_axis(L, tile, overlap):
    """1-D tiling of a length-L axis. Returns (start, end, core_start, core_end); the cores partition [0, L) exactly."""
    if L <= tile: return [(0, L, 0, L)]
    step = tile - overlap; starts = list(range(0, L - tile + 1, step))
    if starts[-1] + tile < L: starts.append(L - tile)
    segs = []
    for i, s in enumerate(starts):
        e = s + tile; c0 = 0 if i == 0 else (starts[i - 1] + tile + s) // 2; c1 = L if i == len(starts) - 1 else (e + starts[i + 1]) // 2
        segs.append((s, e, c0, c1))
    return segs


# --- notebook 04, cell 25: regional_discrepancy --------------------
def regional_discrepancy(density, cell_xy, final, W, H, rc):
    """Compare density-estimated people with ACCEPTED points per region; flag suspicious regions.
    A gap is only a *hint to look again* (density may be wrong, objects may be partially visible) - it never adds people."""
    R = int(rc["region_size"]); out = []
    for iy in range(int(math.ceil(H / R))):
        for ix in range(int(math.ceil(W / R))):
            box = (ix * R, iy * R, min((ix + 1) * R, W), min((iy + 1) * R, H))
            dens = region_count(density, cell_xy, box); n_acc = int(inside_box(final["accepted"], box).sum()); n_unc = int(inside_box(final["uncertain"], box).sum())
            gap = dens - n_acc; reasons = []
            if dens >= rc["min_region_density"] and gap >= max(rc["abs_gap"], rc["rel_gap"] * dens): reasons.append("density>points")
            if n_unc >= rc["min_uncertain"]: reasons.append("many-uncertain")
            if reasons: out.append(dict(key=(ix, iy), box=box, dens=dens, n_acc=n_acc, n_unc=n_unc, gap=gap, reason="+".join(reasons)))
    out.sort(key=lambda r: -max(r["gap"], r["n_unc"])); return out


# --- notebook 04, cell 25: CrowdPredictor --------------------
class CrowdPredictor:
    def __init__(self, model, infer_cfg, device):
        self.model = model.to(device).eval(); self.cfg = copy.deepcopy(infer_cfg); self.dev = device
        self.kind = "point" if model.point else "density"; self.has_density = bool(model.density)
    # ------------------------------------------------------------------ low level
    def _amp(self): return torch.autocast(device_type=self.dev.type, enabled=bool(self.cfg["amp"]) and self.dev.type == "cuda")
    @torch.no_grad()
    def _block(self, x):
        """One forward pass on a padded (1,3,Hb,Wb) tensor -> candidate points / scores (block coords) + density (counts per cell)."""
        with self._amp(): out = self.model(x.to(self.dev))
        den = out["density"][0, 0].float().cpu().numpy() / self.cfg["density_scale"] if "density" in out else None
        if "logits" in out:
            prob = torch.softmax(out["logits"][0].float(), -1)[:, 1]
            pred = out["anchors"].float() + out["offsets"][0].float() * self.model.point_stride; m = prob >= self.cfg["min_candidate_conf"]
            return pred[m].cpu().numpy().astype(np.float32), prob[m].cpu().numpy().astype(np.float32), den
        p, s = density_peaks(den, (8.0, 8.0), self.cfg["peak_min_value"]); return p, s, den
    @torch.no_grad()
    def _run(self, im):
        """im: uint8 RGB image (already resized / normalised). Returns pts (N,2), scores, density in THIS image's pixel frame."""
        h, w = im.shape[:2]; ph, pw = ceil_to(h), ceil_to(w); x = F.pad(to_tensor(im)[None], (0, pw - w, 0, ph - h)); cfg = self.cfg
        if ph * pw <= cfg["max_pixels"]: p, s, d = self._block(x)
        else:                                            # tiled inference: every tile keeps only the points / density cells of its own 'core'
            T, OV = ceil_to(cfg["tile"]), ceil_to(cfg["tile_overlap"]); d = np.zeros((ph // 8, pw // 8), np.float32) if self.has_density else None; P, S = [], []
            for (y0, y1, cy0, cy1) in tile_axis(ph, T, OV):
                for (x0, x1, cx0, cx1) in tile_axis(pw, T, OV):
                    pp, ss, dd = self._block(x[..., y0:y1, x0:x1]); pp = pp + np.array([x0, y0], np.float32)
                    k = (pp[:, 0] >= cx0) & (pp[:, 0] < cx1) & (pp[:, 1] >= cy0) & (pp[:, 1] < cy1); P.append(pp[k]); S.append(ss[k])
                    if d is not None: d[cy0 // 8:cy1 // 8, cx0 // 8:cx1 // 8] = dd[(cy0 - y0) // 8:(cy1 - y0) // 8, (cx0 - x0) // 8:(cx1 - x0) // 8]
            p, s = np.concatenate(P, 0), np.concatenate(S, 0)
        k = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)            # drop anything predicted in the padding
        if d is not None: d = d[:int(math.ceil(h / 8)), :int(math.ceil(w / 8))]
        return p[k], s[k], d
    def _prep(self, img, f):
        h, w = img.shape[:2]
        if abs(f - 1) > 1e-6: img = cv2.resize(img, (max(8, int(round(w * f))), max(8, int(round(h * f)))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        return normalize_exposure(img, self.cfg["exposure_target"]) if self.cfg["exposure_norm"] else img
    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def predict_raw(self, img):
        """Whole-image prediction. All coordinates are returned in ORIGINAL image pixels.
        Returns dict(pts, scores, density (counts/cell or None), cell_xy (original px per density cell), W, H, scale_xy)."""
        H, W = img.shape[:2]; L = max(H, W); f = self.cfg["max_long"] / L if L > self.cfg["max_long"] else (self.cfg["min_long"] / L if L < self.cfg["min_long"] else 1.0)
        im = self._prep(img, f); sx, sy = im.shape[1] / W, im.shape[0] / H; p, s, d = self._run(im)
        return dict(pts=(p / np.array([sx, sy], np.float32)).astype(np.float32), scores=s, density=d, cell_xy=np.array([8.0 / sx, 8.0 / sy], np.float32), W=W, H=H, scale_xy=np.array([sx, sy], np.float32))
    def finalize(self, raw, post=None):
        p = {**{k: self.cfg[k] for k in ("conf_accept", "uncertain_ratio", "dup_radius")}, **(post or {})}
        return finalize_points(raw["pts"], raw["scores"], p["conf_accept"], p["uncertain_ratio"], p["dup_radius"])
    def refine(self, img, raw, post=None, rcfg=None):
        """Regional refinement. Returns (raw2, info). raw2 has an updated candidate pool; `info` lists every region re-analysed."""
        rc = {**self.cfg["refine"], **(rcfg or {})}; p = {**{k: self.cfg[k] for k in ("conf_accept", "uncertain_ratio", "dup_radius")}, **(post or {})}
        pool_p, pool_s, info, done = raw["pts"].copy(), raw["scores"].copy(), [], {}
        if (not rc["enabled"]) or raw.get("density") is None: return dict(raw), info
        for it in range(int(rc["max_iters"])):
            cur = finalize_points(pool_p, pool_s, p["conf_accept"], p["uncertain_ratio"], p["dup_radius"])
            regs = regional_discrepancy(raw["density"], raw["cell_xy"], cur, raw["W"], raw["H"], rc); sc_it = min(rc["scale"] * rc["scale_growth"] ** it, rc["max_scale"])
            todo = [r for r in regs if done.get(r["key"], 0.0) < sc_it - 1e-6][:int(rc["max_regions"])]
            if not todo: break
            for r in todo:
                new_p, new_s, scale_eff = self._reanalyse(img, raw, r, sc_it, rc)
                pool_p, pool_s, action, n_new = self._merge_region(pool_p, pool_s, new_p, new_s, r, p, rc["policy"]); done[r["key"]] = sc_it
                info.append(dict(iteration=it, key=r["key"], box=r["box"], reason=r["reason"], density=r["dens"], n_before=r["n_acc"], n_highres=n_new, action=action, scale=scale_eff))
        return dict(raw, pts=pool_p, scores=pool_s), info
    # ------------------------------------------------------------------ refinement helpers
    def _reanalyse(self, img, raw, r, scale_it, rc):
        """Crop (region + margin) from the ORIGINAL image, up-scale, run the SAME model, map back to original coordinates."""
        x0, y0, x1, y1 = r["box"]; m = int(rc["crop_margin"]); H, W = img.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m); cw, ch = cx1 - cx0, cy1 - cy0
        s_tot = float(raw["scale_xy"].mean()) * scale_it; s_tot = min(s_tot, math.sqrt(rc["max_crop_pixels"] / max(cw * ch, 1)))
        nw, nh = max(32, int(round(cw * s_tot))), max(32, int(round(ch * s_tot))); crop = img[cy0:cy1, cx0:cx1]
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC if s_tot > 1 else cv2.INTER_AREA)
        if self.cfg["exposure_norm"]: crop = normalize_exposure(crop, self.cfg["exposure_target"])
        p, s, _ = self._run(crop); p = crop_to_original(p, cx0, cy0, nw / cw, nh / ch); k = inside_box(p, r["box"])     # keep only the region's own core
        return p[k], s[k], s_tot / float(raw["scale_xy"].mean())
    @staticmethod
    def _merge_region(pool_p, pool_s, new_p, new_s, r, post, policy):
        """Merge high-res candidates for region r into the pool. All policies are coordinate-based; duplicates are removed
        afterwards by the usual confidence-ordered distance NMS in finalize_points().
          union           : add the high-res candidates
          replace_if_more : replace the region's candidates only if the high-res pass ACCEPTS more points than before
          replace         : always replace"""
        k = point_nms(new_p, new_s, post["dup_radius"]); new_p, new_s = new_p[k], new_s[k]; n_new = int((new_s >= post["conf_accept"]).sum())
        if policy == "union": return np.concatenate([pool_p, new_p]), np.concatenate([pool_s, new_s]), "union", n_new
        if policy == "replace" or (policy == "replace_if_more" and n_new > r["n_acc"]):
            keep = ~inside_box(pool_p, r["box"]); return np.concatenate([pool_p[keep], new_p]), np.concatenate([pool_s[keep], new_s]), "replaced", n_new
        return pool_p, pool_s, "kept-original", n_new


# --- notebook 04, cell 25: load_model_checkpoint --------------------
def load_model_checkpoint(path, device, infer_override=None):
    """Load a checkpoint written by Notebook 1. Raises a readable error for missing / invalid files."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: '{path}'.\nTrain a model with Notebook 1 (01_train_evaluate_crowd_localisation.ipynb); it writes <output_dir>/final/final_model.pth. Then set CHECKPOINT_PATH to that file.")
    try: ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try: ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e: raise RuntimeError(f"'{path}' is not a readable PyTorch checkpoint: {e}") from e
    for k in ("state_dict", "spec"):
        if k not in ck: raise RuntimeError(f"Invalid checkpoint '{path}': missing key '{k}' (found {list(ck.keys())[:8]}). Use a file written by Notebook 1.")
    model = build_model(ck["spec"], pretrained=False, freeze_stem=False)
    try: model.load_state_dict(ck["state_dict"])
    except Exception as e: raise RuntimeError(f"Checkpoint weights do not match the architecture in its spec {ck['spec']}: {e}") from e
    infer = copy.deepcopy(DEFAULT_INFER)
    for k, v in (ck.get("infer") or {}).items(): infer[k] = {**infer[k], **v} if isinstance(v, dict) and isinstance(infer.get(k), dict) else v
    for k, v in (infer_override or {}).items(): infer[k] = {**infer[k], **v} if isinstance(v, dict) and isinstance(infer.get(k), dict) else v
    return CrowdPredictor(model, infer, device), ck


# --- notebook 04, cell 26: CountModel --------------------
class CountModel:
    def __init__(self, ckpt_path, device=None, use_refinement=None):

        self.dev = pick_device(device or "auto"); self.pred, self.ck = load_model_checkpoint(ckpt_path, self.dev)
        fin = self.ck.get("final", {}); self.meta = dict(variant=fin.get("variant"), preset=fin.get("preset"), seed=fin.get("seed"), saved=fin.get("saved"), refinement_enabled=fin.get("refinement_enabled"), torch=fin.get("torch"))
        self.refine_on = bool(self.ck.get("infer", {}).get("refine", {}).get("enabled", False)) if use_refinement is None else bool(use_refinement)
        cal = fin.get("calibration"); self.calibration = cal if cal else None

    def predict(self, img_rgb, frame_id=None, refine=None):
        raw = self.pred.predict_raw(img_rgb)
        if (self.refine_on if refine is None else refine): raw, _ = self.pred.refine(img_rgb, raw)
        fin = self.pred.finalize(raw); pts, conf = fin["accepted"], fin["acc_scores"]; dens = raw.get("density"); dsum = float(dens.sum()) if dens is not None else float("nan"); count = int(len(pts)); nunc = int(len(fin["uncertain"]))
        gap = (dsum - count) / max(count, 1) if np.isfinite(dsum) else float("nan")
        unc_level = "n/a" if not np.isfinite(gap) else ("LOW" if abs(gap) < 0.15 and nunc <= 0.1 * max(count, 1) else "MEDIUM" if abs(gap) < 0.35 else "HIGH")
        out = dict(frame_id=frame_id, count=count, head_points=np.asarray(pts, np.float32).reshape(-1, 2), confidence=np.asarray(conf, np.float32), density_map=dens, density_cell_xy=raw.get("cell_xy"),
                   uncertain_points=np.asarray(fin["uncertain"], np.float32).reshape(-1, 2), uncertain_confidence=np.asarray(fin["unc_scores"], np.float32), image_size=(int(raw["W"]), int(raw["H"])),
                   uncertainty=dict(count_density_sum=dsum, n_uncertain=nunc, count_interval=(count, count + nunc), density_gap=float(gap), level=unc_level,
                                    note="heuristic: disagreement between the point count and the density-map sum; not a calibrated probability"))
        if self.calibration: out["count_calibrated"] = float(np.interp(count, self.calibration["x"], self.calibration["y"]))
        return out


# --- notebook 04, cell 27: file_fingerprint --------------------
def file_fingerprint(path):
    """Content hash of a checkpoint, so cached predictions can never be reused after the model is retrained."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()[:10]
