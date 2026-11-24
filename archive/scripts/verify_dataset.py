"""Read-only complete angle audit + eight real-recording preprocessing checks.

Optional CPU no-grad forward checks are not training or diagnostic performance.
Run from the package root; no audio playback, hardware access, or raw writes.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ev3dar.common import load_manifest, read_json, data_path, digest, write_json, read_csv
from ev3dar.features import angle_track, extract_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--prepared', type=Path, default=Path('prepared'))
    parser.add_argument('--report', type=Path, default=Path('validation/actual_data_check.json'))
    parser.add_argument('--forward', action='store_true', help='Random weights, CPU no-grad only; no performance claim.')
    args = parser.parse_args()
    records = load_manifest(args.prepared)
    cfg = read_json(args.prepared/'config.resolved.json')
    errors = []; durations = []; classes = Counter(); samples = {}; small_backsteps = 0
    for r in records:
        try:
            wav = data_path(args.data_root, r['wav'])
            meta = sf.info(wav)
            if meta.samplerate != 48000 or meta.channels != 1:
                raise ValueError('Expected mono48kHz.')
            if abs(meta.duration-r['duration']) > .002:
                raise ValueError('Duration differs from info CSV.')
            angle_path = data_path(args.data_root, r['angles'])
            angle_track(angle_path, meta.duration)
            angles = np.array([float(row['angle_deg']) for row in read_csv(angle_path)])
            small_backsteps += int(np.any(np.diff(angles) < 0))
            durations.append(meta.duration); classes.update(r['classes'])
            samples.setdefault((r['fault_count'], r['band']), r)
        except Exception as exc:
            errors.append(dict(uid=r['uid'], reason=str(exc)))
    checks = []
    if not errors:
        for key, r in sorted(samples.items()):
            before = {k: digest(data_path(args.data_root, r[k])) for k in ['wav','info','angles']}
            features = extract_record(args.data_root, r, cfg)
            after = {k: digest(data_path(args.data_root, r[k])) for k in before}
            if before != after or any(before[k] != r['input_fingerprint'][k]['sha256'] for k in before):
                raise AssertionError('Dataset fingerprint mismatch: '+r['uid'])
            if not all(np.isfinite(v).all() for v in features.values()):
                raise AssertionError('Nonfinite feature: '+r['uid'])
            result = dict(uid=r['uid'], stratum=list(key), raw_unchanged=True,
                shapes={k:list(v.shape) for k,v in features.items()},
                valid_view_seconds=features['view_durations'].tolist())
            if args.forward:
                import torch
                from ev3dar.models import LocalizationCNN, SoundTypeCNN
                torch.set_num_threads(2)
                torch.manual_seed(42)
                with torch.no_grad():
                    ssl = LocalizationCNN(features['ssl'].shape[0]).eval()
                    ids = SoundTypeCNN(use_rpm=cfg['use_rpm']).eval()
                    counts, doa = ssl(torch.from_numpy(features['ssl'])[None])
                    logits, latent = ids(*(torch.from_numpy(features[k])[None]
                        for k in ['id_views','id_masks','view_durations','rpm']))
                    assert all(torch.isfinite(v).all() for v in [counts,doa,logits,latent])
                    result['untrained_forward_shapes'] = dict(count=list(counts.shape),
                        doa=list(doa.shape),classes=list(logits.shape),latent=list(latent.shape))
            checks.append(result)
            print('Read-only sample checked: '+r['uid'], flush=True)
    report = dict(recordings_checked=len(records), angle_and_audio_header_errors=errors,
        wav_seconds_range=[min(durations),max(durations)] if durations else None,
        records_with_small_angle_backsteps=small_backsteps, by_source_class=dict(classes),
        real_sample_checks=checks, training_performed=False, backward_or_optimizer_steps=0,
        forward_uses_random_untrained_weights=bool(args.forward),
        diagnostic_performance_measured=False, dataset_opened_read_only=True)
    write_json(args.report, report)
    print('Report: '+str(args.report), flush=True)
    if errors:
        raise SystemExit('Dataset has invalid records. Review report; do not silently drop or repair.')


if __name__ == '__main__':
    main()
