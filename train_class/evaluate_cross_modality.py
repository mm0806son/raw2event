"""
Cross-modal evaluation.

Supports:
    - QKFormer
    - MobileNetV2
    - a single train_modality -> eval_modality pair
    - the full 3x3 modality matrix in one run
    - reusing a dv run's split_info.json as the canonical validation split
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.data


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
QKFORMER_DIR = os.path.join(SCRIPT_DIR, 'QKFormer', 'cifar10-dvs')

for path in (SCRIPT_DIR, ROOT_DIR, QKFORMER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import neuron as sj_neuron

from train_utils.dataset import EventNpzDataset
from train_utils.mobile_dataset import EventCountImageDataset


VALID_MODALITIES = ('dv', 'raw', 'rgb')


def _patch_lif_backend_when_cupy_missing():
    """
    QKFormer upstream hardcodes backend='cupy' in many MultiStepLIFNode calls.
    When CuPy is unavailable, keep evaluation runnable by falling back to torch.
    """
    if getattr(sj_neuron, "cupy", None) is not None:
        return

    if getattr(sj_neuron.MultiStepLIFNode, "_raw2event_backend_patched", False):
        return

    original_init = sj_neuron.MultiStepLIFNode.__init__

    def _patched_init(self, *args, **kwargs):
        if kwargs.get("backend") == "cupy":
            kwargs["backend"] = "torch"
        return original_init(self, *args, **kwargs)

    sj_neuron.MultiStepLIFNode.__init__ = _patched_init
    sj_neuron.MultiStepLIFNode._raw2event_backend_patched = True
    print("  CuPy unavailable; MultiStepLIFNode backend falls back to torch.")


_patch_lif_backend_when_cupy_missing()


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        k = min(k, output.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def parse_mapping_args(items: List[str]) -> OrderedDict:
    mapping = OrderedDict()
    for item in items:
        if '=' not in item:
            raise ValueError(f"malformed --run entry '{item}'; expected modality=path")
        modality, path = item.split('=', 1)
        modality = modality.strip()
        path = path.strip()
        if modality not in VALID_MODALITIES:
            raise ValueError(f"unsupported modality '{modality}'")
        mapping[modality] = path
    if not mapping:
        raise ValueError("at least one --run modality=path is required")
    return mapping


def sample_basename_from_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    for modality in VALID_MODALITIES:
        suffix = f'_filtered_{modality}.npz'
        if basename.endswith(suffix):
            return basename[:-len(suffix)]
    raise ValueError(f"cannot strip the modality suffix from: {basename}")


def resolve_checkpoint_from_run(run_or_ckpt: str, train_modality: str, checkpoint_tag: str) -> str:
    if os.path.isfile(run_or_ckpt):
        return run_or_ckpt

    pattern = os.path.join(run_or_ckpt, f'output_{train_modality}_*_{checkpoint_tag}.pth')
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"multiple checkpoints matched in '{run_or_ckpt}': {matches}")

    fallback = sorted(glob.glob(os.path.join(run_or_ckpt, f'output_*_{checkpoint_tag}.pth')))
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) > 1:
        raise FileExistsError(f"multiple fallback checkpoints matched in '{run_or_ckpt}': {fallback}")

    raise FileNotFoundError(f"no checkpoint with tag={checkpoint_tag} in '{run_or_ckpt}'")


def resolve_split_source(split_source: str) -> Tuple[str, str]:
    if os.path.isfile(split_source) and split_source.endswith('.json'):
        return os.path.dirname(split_source), split_source

    if os.path.isfile(split_source) and split_source.endswith('.pth'):
        run_dir = os.path.dirname(split_source)
    else:
        run_dir = split_source

    split_info_path = os.path.join(run_dir, 'split_info.json')
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"split_info.json not found: {split_info_path}")
    return run_dir, split_info_path


def load_checkpoint_args(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    return checkpoint.get('args', {})


def resolve_data_dir(cli_data_dir: str, fallback_checkpoint_path: str = '') -> str:
    if cli_data_dir:
        return cli_data_dir
    if fallback_checkpoint_path:
        args = load_checkpoint_args(fallback_checkpoint_path)
        data_dir = args.get('data_dir', '')
        if data_dir:
            return data_dir
    raise ValueError("data_dir is missing; pass --data_dir explicitly")


def canonical_eval_basenames(split_info_path: str, split_data_dir: str) -> List[str]:
    with open(split_info_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    dv_files = sorted(glob.glob(os.path.join(split_data_dir, '*_filtered_dv.npz')))
    if not dv_files:
        raise FileNotFoundError(f"no dv npz files under '{split_data_dir}'")

    test_indices = split_info['test_indices']
    basenames = []
    for idx in test_indices:
        if idx < 0 or idx >= len(dv_files):
            raise IndexError(
                f"split_info index {idx} exceeds the {len(dv_files)} dv files found; "
                f"check that split_source and split_data_dir refer to the same corpus."
            )
        basenames.append(sample_basename_from_filename(dv_files[idx]))
    return basenames


def modality_subset_indices(data_dir: str, eval_modality: str, basenames: List[str]) -> Tuple[List[int], List[str], List[str]]:
    modality_files = sorted(glob.glob(os.path.join(data_dir, f'*_filtered_{eval_modality}.npz')))
    file_map = {sample_basename_from_filename(path): idx for idx, path in enumerate(modality_files)}

    matched_indices = []
    matched_basenames = []
    missing_basenames = []
    for base in basenames:
        idx = file_map.get(base)
        if idx is None:
            missing_basenames.append(base)
        else:
            matched_indices.append(idx)
            matched_basenames.append(base)
    return matched_indices, matched_basenames, missing_basenames


def build_qkformer_model(checkpoint_args: dict, num_classes: int):
    import model as _qkformer_model  # noqa: F401
    from model import vit_snn

    return vit_snn(
        patch_size=16,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=1,
        in_channels=2,
        num_classes=num_classes,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
    )


def build_mobilenet_model(checkpoint_args: dict, num_classes: int):
    from train_mobileNetV2 import MobileNetV2Classifier

    # Replay the input channel count that the ckpt was trained with:
    #   stacked_histogram → 2 * rep_T (e.g. 20 for the default T=10)
    #   timestack         → 3 (RGB replicate, project default)
    # Older ckpts predate these args; fall back to legacy 3-channel.
    representation = checkpoint_args.get('representation', 'timestack')
    if representation == 'stacked_histogram':
        rep_T = checkpoint_args.get('rep_T', 10)
        in_channels = 2 * rep_T
    else:
        in_channels = checkpoint_args.get('output_channels', 3)

    return MobileNetV2Classifier(
        num_classes=num_classes,
        pretrained=False,
        dropout=checkpoint_args.get('dense_dropout', 0.5),
        hidden_dim=checkpoint_args.get('hidden_dim', 128),
        in_channels=in_channels,
    )


def set_module_backend(module: nn.Module, backend: str):
    for child in module.modules():
        if hasattr(child, 'backend'):
            child.backend = backend


def build_dataset(model_family: str, data_dir: str, eval_modality: str, checkpoint_args: dict):
    if model_family == 'qkformer':
        return EventNpzDataset(
            data_dir=data_dir,
            modality=eval_modality,
            T=checkpoint_args.get('T', 16),
            target_h=128,
            target_w=128,
        )

    if model_family == 'mobilenetv2':
        # Dispatch on the representation recorded in the ckpt's args, so
        # stacked_histogram ckpts get the matching dataset class (and
        # therefore the correct input channel count). Older ckpts predate
        # this field — default to timestack for backward compatibility.
        representation = checkpoint_args.get('representation', 'timestack')
        if representation == 'stacked_histogram':
            from train_utils.mobile_dataset import EventStackedHistogramDataset
            return EventStackedHistogramDataset(
                data_dir=data_dir,
                modality=eval_modality,
                T=checkpoint_args.get('rep_T', 10),
                target_h=checkpoint_args.get('target_h', 128),
                target_w=checkpoint_args.get('target_w', 128),
                output_size=checkpoint_args.get('input_size', 224),
                count_cutoff=checkpoint_args.get('rep_count_cutoff', 10),
                normalize=True,
            )
        return EventCountImageDataset(
            data_dir=data_dir,
            modality=eval_modality,
            target_h=checkpoint_args.get('target_h', 128),
            target_w=checkpoint_args.get('target_w', 128),
            output_channels=checkpoint_args.get('output_channels', 3),
            output_size=checkpoint_args.get('input_size', 224),
            normalize=True,
        )

    raise ValueError(f"unsupported model family: {model_family}")


def evaluate_loader(model_family: str, model: nn.Module, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            loss_meter.update(loss.item(), batch_size)
            acc1_meter.update(acc1.item(), batch_size)
            acc5_meter.update(acc5.item(), batch_size)

            if model_family == 'qkformer':
                functional.reset_net(model)

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def evaluate_pair(
    model_family: str,
    checkpoint_path: str,
    train_modality: str,
    eval_modality: str,
    data_dir: str,
    canonical_basenames: List[str],
    batch_size: int,
    workers: int,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint_args = checkpoint.get('args', {})
    num_classes = checkpoint_args.get('num_classes', 10)

    dataset = build_dataset(model_family, data_dir, eval_modality, checkpoint_args)
    subset_indices, matched_basenames, missing_basenames = modality_subset_indices(data_dir, eval_modality, canonical_basenames)
    if not subset_indices:
        raise ValueError(f"{train_modality}->{eval_modality} has no evaluable samples")

    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
        pin_memory=(device.type == 'cuda'),
    )

    if model_family == 'qkformer':
        model = build_qkformer_model(checkpoint_args, num_classes=num_classes)
        if device.type != 'cuda':
            set_module_backend(model, 'torch')
    else:
        model = build_mobilenet_model(checkpoint_args, num_classes=num_classes)

    model.load_state_dict(checkpoint['model'])
    model.to(device)

    loss, acc1, acc5 = evaluate_loader(model_family, model, loader, device)

    return {
        'model_family': model_family,
        'train_modality': train_modality,
        'eval_modality': eval_modality,
        'checkpoint_path': checkpoint_path,
        'loss': loss,
        'acc1': acc1,
        'acc5': acc5,
        'requested_eval_samples': len(canonical_basenames),
        'matched_eval_samples': len(matched_basenames),
        'missing_eval_samples': len(missing_basenames),
        'missing_basenames': missing_basenames,
    }


def evaluate_loader_per_sample(model_family: str, model: nn.Module, data_loader, device):
    """Like evaluate_loader but returns per-sample correctness + predictions
    instead of aggregate meters. Used by paired-bootstrap CI tooling
    (tools/v2e_baseline/cross_modal_eval_with_ci.py)."""
    model.eval()
    correct: List[int] = []
    preds: List[int] = []
    targets: List[int] = []
    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device, non_blocking=True).float()
            target_dev = target.to(device, non_blocking=True)
            output = model(images)
            pred = output.argmax(dim=1)
            ok = (pred == target_dev).to(torch.int8).cpu().tolist()
            correct.extend(ok)
            preds.extend(pred.cpu().tolist())
            targets.extend(target.cpu().tolist())
            if model_family == 'qkformer':
                functional.reset_net(model)
    return correct, preds, targets


def run_inference_on_indices(
    ckpt_path: str,
    data_dir: str,
    test_indices: List[int],
    device: str = 'cuda:0',
    batch_size: int = 32,
    workers: int = 4,
    model_family: str = 'qkformer',
    test_modality: str = 'dv',
):
    """Helper used by tools/v2e_baseline/cross_modal_eval_with_ci.py.

    Loads ckpt_path, builds a Dataset over data_dir for test_modality,
    selects samples by absolute index test_indices, runs inference, and
    returns per-sample correctness for paired bootstrap CI.

    Returns dict with keys: correct (List[int 0/1]), preds, targets, acc.
    """
    dev = torch.device(device if (torch.cuda.is_available() and 'cuda' in device) else 'cpu')

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    checkpoint_args = checkpoint.get('args', {})
    num_classes = checkpoint_args.get('num_classes', 10)

    dataset = build_dataset(model_family, data_dir, test_modality, checkpoint_args)
    subset = torch.utils.data.Subset(dataset, list(test_indices))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
        pin_memory=(dev.type == 'cuda'),
    )

    if model_family == 'qkformer':
        model = build_qkformer_model(checkpoint_args, num_classes=num_classes)
        if dev.type != 'cuda':
            set_module_backend(model, 'torch')
    else:
        model = build_mobilenet_model(checkpoint_args, num_classes=num_classes)
    model.load_state_dict(checkpoint['model'])
    model.to(dev)

    correct, preds, targets = evaluate_loader_per_sample(model_family, model, loader, dev)
    acc = float(sum(correct)) / max(len(correct), 1)
    return {
        'correct': correct,
        'preds': preds,
        'targets': targets,
        'acc': acc,
        'n': len(correct),
    }


def save_results(results: List[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(output_dir, f'cross_modal_eval_{timestamp}.json')
    csv_path = os.path.join(output_dir, f'cross_modal_eval_{timestamp}.csv')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        'model_family',
        'train_modality',
        'eval_modality',
        'checkpoint_path',
        'split_source',
        'data_dir',
        'loss',
        'acc1',
        'acc5',
        'requested_eval_samples',
        'matched_eval_samples',
        'missing_eval_samples',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})

    return json_path, csv_path


def print_summary(results: List[dict]):
    print('\n' + '=' * 88)
    print('Cross-Modality Evaluation Summary')
    print('=' * 88)
    print(f"{'Train':<8} {'Eval':<8} {'Acc@1':>8} {'Acc@5':>8} {'Loss':>10} {'Matched/Req':>14} {'Missing':>8}")
    for row in results:
        matched_req = f"{row['matched_eval_samples']}/{row['requested_eval_samples']}"
        print(
            f"{row['train_modality']:<8} {row['eval_modality']:<8} "
            f"{row['acc1']:>8.2f} {row['acc5']:>8.2f} {row['loss']:>10.4f} "
            f"{matched_req:>14} "
            f"{row['missing_eval_samples']:>8}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-modal evaluation')
    parser.add_argument('--model_family', choices=['qkformer', 'mobilenetv2'], required=True)
    parser.add_argument(
        '--run',
        action='append',
        default=[],
        help='map a training modality to a run directory or checkpoint: modality=/path/to/run',
    )
    parser.add_argument('--eval_modalities', nargs='+', default=list(VALID_MODALITIES), choices=VALID_MODALITIES)
    parser.add_argument('--split_source', required=True, help='a dv run directory, a split_info.json, or a checkpoint')
    parser.add_argument('--data_dir', default='', help='directory holding the NPZ files for all three modalities')
    parser.add_argument('--split_data_dir', default='', help='NPZ directory used to rebuild the dv validation split; defaults to --data_dir')
    parser.add_argument('--checkpoint_tag', choices=['best', 'latest'], default='best')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--output_dir', default=os.path.join(SCRIPT_DIR, 'output'))
    return parser.parse_args()


def main(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA unavailable; falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)

    run_mapping = parse_mapping_args(args.run)
    resolved_checkpoints = OrderedDict()
    for train_modality, path in run_mapping.items():
        resolved_checkpoints[train_modality] = resolve_checkpoint_from_run(path, train_modality, args.checkpoint_tag)

    _, split_info_path = resolve_split_source(args.split_source)
    data_dir = resolve_data_dir(args.data_dir, next(iter(resolved_checkpoints.values())))
    split_data_dir = args.split_data_dir or data_dir
    canonical_basenames = canonical_eval_basenames(split_info_path, split_data_dir)

    print(f"📌 model_family={args.model_family}")
    print(f"📌 data_dir={data_dir}")
    print(f"📌 split_source={split_info_path}")
    print(f"📌 canonical dv validation samples={len(canonical_basenames)}")
    print(f"📌 train modalities={list(resolved_checkpoints.keys())}")
    print(f"📌 eval modalities={args.eval_modalities}")

    results = []
    for train_modality, checkpoint_path in resolved_checkpoints.items():
        for eval_modality in args.eval_modalities:
            print(f"\n🔍 Evaluating {train_modality} -> {eval_modality}")
            result = evaluate_pair(
                model_family=args.model_family,
                checkpoint_path=checkpoint_path,
                train_modality=train_modality,
                eval_modality=eval_modality,
                data_dir=data_dir,
                canonical_basenames=canonical_basenames,
                batch_size=args.batch_size,
                workers=args.workers,
                device=device,
            )
            result['split_source'] = split_info_path
            result['data_dir'] = data_dir
            results.append(result)
            print(
                f"  Acc@1={result['acc1']:.2f} Acc@5={result['acc5']:.2f} "
                f"Loss={result['loss']:.4f} Matched={result['matched_eval_samples']}/{result['requested_eval_samples']} "
                f"Missing={result['missing_eval_samples']}"
            )

    json_path, csv_path = save_results(results, args.output_dir)
    print_summary(results)
    print(f"\n💾 JSON: {json_path}")
    print(f"💾 CSV : {csv_path}")


if __name__ == '__main__':
    main(parse_args())
