"""Fine-tune the CoR classifier head on labeled tomographic slices.

Single-GPU only; no distributed, no AMP, no wandb.

NOTE: only the `nn.Linear(embed_dim, 2)` head is trained — the vendored
`ClassificationModel.forward` hardcodes `self.model.eval()` + `torch.no_grad()`
around the DINOv2 backbone, so backbone gradients never flow even if you set
`requires_grad=True`. Full backbone fine-tuning is out of scope for this repo.

Data layout:
    LABELS_DIR/centered/*.tif        # well-centered reconstructions  (label 1)
    LABELS_DIR/off_centered/*.tif    # off-centered reconstructions   (label 0)
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from tomo_center import logging as tca_logging
from tomo_center.ai.model_archs import ClassificationModel, _make_dinov2_model

log = tca_logging.getLogger(__name__)

_TIFF_EXTS = (".tif", ".tiff")


# ---------- dataset -----------------------------------------------------------

def random_crop(img_gts, gt_patch_size, patch_corner = None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        
        gt_patch_size (int): GT patch size.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """
    if not isinstance(gt_patch_size,list):
        gt_patch_size = [gt_patch_size,gt_patch_size]
    if not isinstance(img_gts, list):
        img_gts = [img_gts]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_gt, w_gt = img_gts[0].shape[0:2]

    if h_gt < gt_patch_size[0]:

        top_pad_gt = gt_patch_size[0] - h_gt
        if input_type == 'Tensor':
            img_gts = [torch.nn.functional.pad(v,(0,0,0,top_pad_gt),'reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            pad_size = tuple([(0,top_pad_gt) if d == 0 else (0,0) for d in range(img_gts[0].ndim)])
            img_gts = [np.pad(v,pad_size,mode='reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].shape[0:2]


    if w_gt < gt_patch_size[1]:

        left_pad_gt = gt_patch_size[1] - w_gt
        if input_type == 'Tensor':
            img_gts = [torch.nn.functional.pad(v,(0,left_pad_gt,0,0),'reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            pad_size = tuple([(0,left_pad_gt) if d == 1 else (0,0) for d in range(img_gts[0].ndim)])
            img_gts = [np.pad(v,pad_size,mode='reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].shape[0:2]

    # randomly choose top and left coordinates for lq patch
    if patch_corner is None:
        top_gt = random.randint(0, h_gt - gt_patch_size[0])
        left_gt = random.randint(0, w_gt - gt_patch_size[1])
    else:
        assert len(patch_corner) == 2
        top_gt = patch_corner[0]
        left_gt = patch_corner[1]
        if (top_gt < 0) | (top_gt > (h_gt - gt_patch_size[0])):
            top_gt = random.randint(0, h_gt - gt_patch_size[0])
        if (left_gt < 0) | (left_gt > (w_gt - gt_patch_size[1])):
            left_gt = random.randint(0, w_gt - gt_patch_size[1])

    # crop corresponding gt patch
    
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1]] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1], ...] for v in img_gts]


    if len(img_gts) == 1:
        img_gts = img_gts[0]
    
    return img_gts

def sample_patch_corner(sample_patch_probs,mask,window_size,num_windows):
    grid_indices = np.where(np.random.multinomial(1,sample_patch_probs/sample_patch_probs.sum(),num_windows))[1]
    patch_corners = []
    for grid_idx in grid_indices:
        grid_idx_ = []
        img_grids = np.indices(mask.shape)
        for d in range(len(list(mask.shape))):
            grid_idx_.append(img_grids[d].reshape((-1,1)).squeeze()[grid_idx])
        if grid_idx_[-1] == 0:
            grid_idx_ = grid_idx_[:-1]
        patch_corner = [grid_idx_[i]-window_size//2 for i in range(len(grid_idx_))]
        patch_corner = [max(0, pc) for pc in patch_corner]
        patch_corner = [min(pc, mask.shape[i] - window_size - 1) for i, pc in enumerate(patch_corner)]
        patch_corner = tuple(patch_corner)
        patch_corners.append(patch_corner)
    
    return patch_corners

def _collect_pairs(image_root,meta_info_file,enlarge_factor,split_kw:str='case'):
    if type(split_kw) is not str:
        log.error("Input argumet: split_kw is expected to be of type str. Got %s instead.",type(split_kw).__name__)
        raise TypeError("Unexpected type for input argument.")
    if split_kw not in ['case','file']:
        log.error("Data splitting currently only supported at file and case levels. Got %s instead.",split_kw)
        raise ValueError("Unexpected input %s",split_kw)
    
    if type(image_root) is str:
        image_root = [image_root]
    if type(meta_info_file) is str:
        meta_info_file = [meta_info_file]
    if type(enlarge_factor) is str:
        enlarge_factor = [enlarge_factor]
    
    if not isinstance(image_root,list):
        log.error("The root directory is expected to be a list. Got %s instead.",type(image_root).__name__)
        raise TypeError("Unexpected type for input argument.")
    if not isinstance(meta_info_file,list):
        log.error("The path to the meta data is expected to be a list. Got %s instead.",type(meta_info_file).__name__)
        raise TypeError("Unexpected type for input argument.")
    if not isinstance(enlarge_factor,list):
        log.error("The enlarge factor is expected to be a list. Got %s instead.", type(enlarge_factor).__name__)
        raise TypeError("Unexpected type for input argument.")
    
    if not len(image_root) == len(meta_info_file) == len(enlarge_factor):
        log.error("Numbers of root directories, meta data files, and enlarge factors do not match. Got %d, %d, and %d, respectively.",len(image_root),len(meta_info_file),len(enlarge_factor))
        raise ValueError("Numbers of root directories, meta data files, and enlarge factors do not match.")

    pairs = []
    tomo_masks = {}
    split_values = []
    for image_root_, meta_info_file_, enlarge_factor_ in zip(image_root,meta_info_file,enlarge_factor):
        with open(meta_info_file_,'r') as fin:
            for line in fin:
                metadata = line.strip().split(' ')
                image_dir = metadata[0]
                optimal_cor = float(metadata[1])

                image_files_ = sorted(list((Path(image_root_) / image_dir).glob('*.tiff')))
                cors = [extract_cor_from_filename(str(image_file)) for image_file in image_files_]
                labels = [cor==optimal_cor for cor in cors]
                if not np.any(np.array(labels)):
                    log.warning("Case %s does not contain any images with the actual cor.",image_dir)
                
                row, col = metadata[2][1:-1].split(',')
                row, col = int(row), int(col)
                if row != col:
                    log.warning("Images from case %s have %d rows and %d columns. To skip...",image_dir,row,col)
                    continue
                
                if row not in list(tomo_masks.keys()):
                    x_coords, y_coords = np.meshgrid(np.arange(col)-(col-1)/2, np.arange(row)-(row-1)/2, indexing='xy')
                    mask = (x_coords**2+y_coords**2) <= ((row-1) / 2)**2
                    tomo_masks[row] = mask
                
                pairs.extend([(p,l) for p,l in zip(image_files_,labels)] * enlarge_factor_)
                if split_kw == 'case':
                    split_values.extend([image_dir for p in image_files_] * enlarge_factor_)
    if split_kw == 'file':
        split_values = list(range(len(pairs)))
        return pairs, tomo_masks, split_values
    else:
        return pairs, tomo_masks, split_values

def extract_cor_from_filename(filename: str) -> float:
    """Extract COR value from filename"""
    import os
    import re
    base = os.path.splitext(os.path.basename(filename))[0]

    # Pattern: center604.50
    match = re.search(r'center(\d+)\.(\d+)', base)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")

    match = re.search(r'center(\d+)_(\d+)', base)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")

    match = re.search(r'center(\d+)', base)
    if match:
        return float(match.group(1))
    
    match = re.search(r'_(\d+)\.(\d+)', base)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")

    return None

class CoRDataset(Dataset):
    """Yields (image_tensor, label).

    Tensor shape: (1, 1, sz, sz) — matches what `ClassificationModel` expects
    when indexed as `sample['images'][:, 0]` in the single-window branch.
    """

    def __init__(self, pairs: List[Tuple[Path,int]], window_size: int, num_windows: int, augment: bool, tomo_masks: Dict[str,np.ndarray]):
        self.pairs = list(pairs)
        self.window_size = window_size
        self.num_windows = num_windows
        self.augment = augment
        self.tomo_masks = tomo_masks

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, label = self.pairs[idx]
        img = tifffile.imread(str(path))
        if img.ndim != 2:
            raise ValueError(f"{path.name}: expected 2D image, got {img.shape}")
        img = img.astype(np.float32, copy=False)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        h, w = img.shape
        sz = self.window_size
        if h < sz or w < sz:
            raise ValueError(f"{path.name}: image {img.shape} smaller than window {sz}")

        mask = self.tomo_masks[h]

        if self.augment and random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        sample_patch_probs = (mask / mask.sum()).reshape((-1,1)).squeeze().astype(np.float64)
        patch_corners = sample_patch_corner(sample_patch_probs,mask,self.window_size,self.num_windows)
        crop = [random_crop(img, self.window_size, patch_corner = patch_corner) for patch_corner in patch_corners]

        # (sz, sz) -> (k, 1, sz, sz): (channel=1, window-index slot used as channel-of-window)
        tensor = torch.concat([torch.from_numpy(np.ascontiguousarray(c)).float().unsqueeze(0).unsqueeze(0) for c in crop],dim=0)
        return tensor, int(label)


def _split_pairs(pairs, val_split: float, seed: int, split_values=None):
    rng = random.Random(seed)
    pairs = list(pairs)
    if split_values is None:
        rng.shuffle(pairs)
        n_val = max(1, int(round(len(pairs) * val_split))) if val_split > 0 else 0
        return pairs[n_val:], pairs[:n_val]
    else:
        if not isinstance(split_values,list):
            log.error("Indices for train/validation data splitting expected to be a list. Got %s instead.",type(split_values).__name__)
            raise TypeError("Unexpected type for input argument.")
        
        common_values = list(set(split_values))
        rng.shuffle(common_values)
        n_val = max(1, int(round(len(common_values)*val_split))) if val_split > 0 else 0
        train_common_values, val_common_values = common_values[n_val:], common_values[:n_val]
        train_split_indices = [idx for idx,val in enumerate(split_values) if val in train_common_values]
        val_split_indices = [idx for idx,val in enumerate(split_values) if val in val_common_values]
        return [pairs[i] for i in train_split_indices],[pairs[i] for i in val_split_indices]

def _resample_pairs(pairs, seed: int, resampling_method:str="upsample"):
    if type(resampling_method) is not str:
        log.error("Input argumet: resampling_method is expected to be of type str. Got %s instead.",type(resampling_method).__name__)
        raise TypeError("Unexpected type for input argument.")
    rng = random.Random(seed)
    pairs_pos = [p for p in pairs if p[1]]
    
    pairs_neg = [p for p in pairs if not p[1]]
    

    if len(pairs_pos) > len(pairs_neg):
        pairs_large = pairs_pos
        pairs_small = pairs_neg
    else:
        pairs_large = pairs_neg
        pairs_small = pairs_pos
    
    if resampling_method == "upsample":
        pairs_small_resample = rng.choices(pairs_small,k=len(pairs_large))
        pairs_resample = pairs_large+pairs_small_resample
        rng.shuffle(pairs_resample)
        return pairs_resample
    elif resampling_method == 'downsample':
        pairs_large_resample = rng.sample(pairs_large,k=len(pairs_small))
        pairs_resample = pairs_large_resample+pairs_small
        rng.shuffle(pairs_resample)
        return pairs_resample
    else:
        log.error(f"Resampling method {resampling_method} currently not supported. Please choose among: upsample, downsample, or undo resampling with: none")
        raise ValueError("Unexpected input %s",resampling_method)


# ---------- model build / load ------------------------------------------------

def _build_model(args, device: torch.device) -> ClassificationModel:
    backbone = _make_dinov2_model()
    if args.resume is None:
        log.info("Loading backbone from torch.hub (%s) — requires internet.", args.base_model)
        try:
            hub_model = torch.hub.load("facebookresearch/dinov2", args.base_model)
        except Exception as e:
            raise SystemExit(
                f"--resume not given. Loading {args.base_model} from torch.hub requires "
                f"internet (failed: {e}). Either pass --resume <existing checkpoint.pt> "
                f"or run on a machine with internet access first."
            ) from e
        backbone.load_state_dict(hub_model.state_dict(), strict=False)

    if args.freeze_backbone_ok:
        for p in backbone.parameters():
            p.requires_grad = False
    multi_instances = (args.num_windows>1)
    model = ClassificationModel(
        backbone,
        embed_dim=backbone.embed_dim,
        num_windows=[args.num_windows],
        multi_instances=multi_instances,
        freeze_backbone_ok=args.freeze_backbone_ok
    )

    if args.freeze_pooler_ok:
        for p in model.attention.parameters():
            p.requires_grad = False
        for p in model.gate.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = False

    if args.resume is not None:
        log.info("Resuming full classifier from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        states = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        states = {(k.replace("module.", "") if k.startswith("module.") else k): v
                  for k, v in states.items()}
        msg = model.load_state_dict(states, strict=False)
        if msg.missing_keys:
            log.warning("Missing keys when loading --resume: %d (showing first 5: %s)",
                        len(msg.missing_keys), msg.missing_keys[:5])
        if msg.unexpected_keys:
            log.warning("Unexpected keys when loading --resume: %d (showing first 5: %s)",
                        len(msg.unexpected_keys), msg.unexpected_keys[:5])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("Model built: %d trainable params / %d total.",
             n_trainable, n_total)

    model.to(device)
    return model


# ---------- optimizer / scheduler ---------------------------------------------

def _make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no weight decay on gain/bias/norm parameters (matches Polaris script)."""
    def is_no_decay(name: str, p: torch.nn.Parameter) -> bool:
        return p.ndim < 2 or "bias" in name or "ln" in name or "bn" in name

    no_decay, decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if is_no_decay(n, p) else decay).append(p)

    return torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0},
         {"params": decay,    "weight_decay": weight_decay}],
        lr=lr,
    )


def _lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


# ---------- train / eval steps ------------------------------------------------

def _epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None):
    """One pass over `loader`. If optimizer is given, train; otherwise eval."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)     # (B, k, 1, sz, sz)
            labels = labels.to(device, non_blocking=True)
            # ClassificationModel.forward takes a list of dicts (one per scale).
            if train_mode:
                logits = model({"images": imgs})            # (B, 2)
            else:
                logits = model([{"images": imgs}])
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += bs
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# ---------- entry point -------------------------------------------------------

def run_training(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU (slow).")
        args.device = "cpu"
    device = torch.device(args.device)

    log.info("Scanning for labeled TIFFs ...")
    
    pairs, tomo_masks, split_values = _collect_pairs(args.image_root,args.meta_info_file,args.enlarge_factor,args.split_kw)
    log.info("  found %d slices total (centered=%d, off_centered=%d)",
             len(pairs),
             sum(1 for _, y in pairs if y == 1),
             sum(1 for _, y in pairs if y == 0))

    train_pairs, val_pairs = _split_pairs(pairs, args.val_split, args.seed, split_values)
    log.info("  split: train=%d  val=%d", len(train_pairs), len(val_pairs))
    if args.resampling_method != 'none':
        train_pairs = _resample_pairs(train_pairs, args.seed, resampling_method=args.resampling_method)
        log.info("  split after %s: train=%d  val=%d",args.resampling_method, len(train_pairs), len(val_pairs))

    train_ds = CoRDataset(train_pairs, args.window_size, args.num_windows, augment=not args.no_augment,tomo_masks=tomo_masks)
    val_ds = CoRDataset(val_pairs, args.window_size, args.num_windows, augment=False,tomo_masks=tomo_masks) if val_pairs else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
                  if val_ds is not None else None)

    model = _build_model(args, device)
    optimizer = _make_optimizer(model, args.lr, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_lr_lambda(args.warmup_steps, total_steps))
    loss_fn = torch.nn.CrossEntropyLoss()

    args.out.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0  # val_acc if we have val, else -train_loss
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _epoch(model, train_loader, loss_fn, device,
                                       optimizer=optimizer, scheduler=scheduler)
        if val_loader is not None:
            val_loss, val_acc = _epoch(model, val_loader, loss_fn, device)
            log.info("epoch %2d/%d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.3f",
                     epoch, args.epochs, train_loss, train_acc, val_loss, val_acc)
            metric = val_acc
        else:
            log.info("epoch %2d/%d  train_loss=%.4f train_acc=%.3f  (no val)",
                     epoch, args.epochs, train_loss, train_acc)
            metric = -train_loss
        if args.checkpoint_every_epoch:
            torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "val_acc": (metric if val_loader is not None else None),
                    },
                    (args.out / f"epoch_{epoch}.pt"),
                )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_acc": (metric if val_loader is not None else None),
                },
                (args.out / "epoch_best.pt"),
            )
            log.info("  saved best -> %s%s",
                     (args.out / "epoch_best.pt"),
                     f" (val_acc={metric:.3f})" if val_loader is not None else "")

    log.info("Training done. Best epoch=%d. Checkpoint: %s",
             best_epoch, (args.out / 'epoch_best.pt'))
    return 0
