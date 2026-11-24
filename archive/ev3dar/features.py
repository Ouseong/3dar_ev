"""48kHz numeric features. No independent crop loudness normalization/splicing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf
from scipy import signal
from .common import config_hash, data_path, read_csv, write_json, load_manifest


def mel_bank(cfg):
    hz_to_mel = lambda x: 2595.0 * np.log10(1.0 + np.asarray(x) / 700.0)
    mel_to_hz = lambda x: 700.0 * (10.0 ** (np.asarray(x) / 2595.0) - 1.0)
    points = mel_to_hz(np.linspace(hz_to_mel(cfg['fmin']), hz_to_mel(cfg['fmax']), cfg['n_mels']+2))
    freqs = np.fft.rfftfreq(cfg['n_fft'], 1.0/cfg['sample_rate'])
    left = (freqs[None,:]-points[:-2,None])/(points[1:-1,None]-points[:-2,None])
    right = (points[2:,None]-freqs[None,:])/(points[2:,None]-points[1:-1,None])
    weights = np.maximum(0.0, np.minimum(left, right))
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def logmel(audio, cfg):
    n = cfg['n_fft']; fs = cfg['sample_rate']
    if len(audio) < n: audio = np.pad(audio, (0, n-len(audio)))
    _, times, power = signal.spectrogram(audio.astype(np.float64), fs=fs, window='hann',
        nperseg=n, noverlap=n-cfg['hop_length'], detrend=False, scaling='spectrum', mode='psd')
    mel = mel_bank(cfg) @ power
    db = 10.0*np.log10(np.maximum(mel, 10.0**(cfg['db_floor']/10.0)))
    return db.astype(np.float32), times


def angle_track(path, duration):
    rows = read_csv(path)
    t = np.array([float(r['elapsed_sec']) for r in rows])
    a = np.array([float(r['angle_deg']) for r in rows])
    keep = np.isfinite(t) & np.isfinite(a) & (t >= 0) & (t <= duration)
    t, a = t[keep], a[keep]
    if len(t) < 20 or np.any(np.diff(t) < 0): raise ValueError('Invalid angle timestamps.')
    t, unique = np.unique(t, return_index=True); a = a[unique]
    if np.min(a) < 0 or np.max(a) > 360: raise ValueError('Angles outside saved0..360 interval.')
    if np.any(np.diff(a) < -1.0) or np.any(np.diff(a) > 20.0):
        raise ValueError('Unacceptable recorded angle discontinuity; no silent repair.')
    # Gate anchors come from the acquisition's interpolated0/360 boundaries.
    # Keep an increasing subsequence for inverse interpolation; repeated end
    # samples/small(<1deg) backsteps are documented, not invented extra audio.
    increasing = np.r_[True, a[1:] > np.maximum.accumulate(a[:-1])]
    t, a = t[increasing], a[increasing]
    interior = (t > 0) & (t < duration) & (a > 0) & (a < 360)
    return np.r_[0.0, t[interior], duration], np.r_[0.0, a[interior], 360.0]


def circular_segments(start, end):
    if not 0 <= start < 360 or not start < end <= start+360:
        raise ValueError('Invalid angular ROI.')
    return [(start, end)] if end <= 360 else [(start, 360.0), (0.0, end-360.0)]


def ssl_input(db, times, angle_t, angle_a, cfg):
    centers = np.interp(np.arange(360), angle_a, angle_t)
    central = np.stack([np.interp(centers, times, band) for band in db])
    context = float(cfg['ssl_context_seconds'])
    if context <= 0: return central.astype(np.float32)
    # Explicit assumption:1sec context window, stride1angular degree. This is
    # NOT1sec overlap, nor independent11ms FFTs. Edge contexts are truncated;
    # no end/start waveform is fabricated. Retain central features to preserve
    # directional peaks in addition to local-context average spectral power.
    p = np.power(10.0, db.astype(np.float64)/10.0)
    sums = np.c_[np.zeros(len(db)), np.cumsum(p, axis=1)]
    lo = np.searchsorted(times, centers-context/2, side='left')
    hi = np.searchsorted(times, centers+context/2, side='right')
    lo = np.minimum(lo, len(times)-1); hi = np.maximum(hi, lo+1)
    avg = (sums[:,hi]-sums[:,lo])/np.maximum(hi-lo,1)
    local = 10*np.log10(np.maximum(avg, 10**(cfg['db_floor']/10)))
    return np.concatenate([central, local], axis=0).astype(np.float32)


def extract_record(root, record, cfg):
    audio, fs = sf.read(data_path(root, record['wav']), dtype='float32')
    if fs != cfg['sample_rate'] or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono48kHz WAV: '+record['uid'])
    duration = len(audio)/fs
    if abs(duration-record['duration']) > .002: raise ValueError('WAV/metadata duration mismatch.')
    at, aa = angle_track(data_path(root, record['angles']), duration)
    db, times = logmel(audio, cfg)
    full = ssl_input(db, times, at, aa, cfg)
    positions = cfg['positions']; m = cfg['n_mels']; frames = cfg['id_frames']
    views = np.full((len(positions),2,m,frames), cfg['db_floor'], np.float32)
    masks = np.zeros((len(positions),2,frames), np.float32)
    durations = np.zeros((len(positions),2), np.float32)
    rpm_features = np.zeros((len(positions),2), np.float32)
    common_rpm = np.median(np.array(record['rpm_intervals']), axis=0)
    for pi, pos in enumerate(positions):
        mids=[]; weights=[]
        for vi,(a,b) in enumerate(circular_segments(*cfg['rois'][pos])):
            t0,t1 = np.interp([a,b], aa, at)
            segment = audio[round(t0*fs):round(t1*fs)]
            if len(segment) < 2: raise ValueError('Empty angular ROI.')
            part,_ = logmel(segment, cfg)
            n = part.shape[1]
            if n > frames:
                # Small motor-speed variability: resample FEATURE time only,
                # never alter source audio playback, labels, or RPM.
                part = np.stack([np.interp(np.linspace(0,n-1,frames),np.arange(n),v) for v in part])
                n = frames
            views[pi,vi,:,:n] = part; masks[pi,vi,:n] = 1
            durations[pi,vi] = t1-t0
            mids.append(np.interp((t0+t1)/2, [0,duration], common_rpm)); weights.append(t1-t0)
        mean=np.average(mids,weights=weights); spread=np.sqrt(np.average((np.asarray(mids)-mean)**2,weights=weights))
        rpm_features[pi]=[mean/10000.0,spread/10000.0]
    return dict(ssl=full,id_views=views,id_masks=masks,view_durations=durations,rpm=rpm_features)


def feature_key(record, cfg):
    keys=['sample_rate','n_fft','hop_length','n_mels','fmin','fmax','db_floor',
          'angular_bins','ssl_context_seconds','id_frames','positions','rois']
    feature_cfg={k:cfg[k] for k in keys}
    return hashlib.sha256(json.dumps(dict(version=1, uid=record['uid'], cfg=feature_cfg,
        inputs=record['input_fingerprint']), sort_keys=True).encode()).hexdigest()


def cache_path(out, uid):
    if not uid.replace('_','').isalnum(): raise ValueError('Invalid cache UID.')
    return Path(out)/'cache'/(uid+'.npz')


def build_cache(root, out, cfg, records=None):
    records = records if records is not None else load_manifest(out)
    (Path(out)/'cache').mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records,1):
        p=cache_path(out,r['uid']);key=feature_key(r,cfg)
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                if str(z['feature_key'].item())==key:continue
        feat=extract_record(root,r,cfg)
        temporary=p.with_suffix('.pending.npz');np.savez_compressed(temporary,feature_key=np.array(key),**feat);temporary.replace(p)
        if i%25==0 or i==len(records):print(f'Features {i}/{len(records)}',flush=True)
    write_json(Path(out)/'cache/config.json', dict(config_hash=config_hash(cfg), feature_version=1))


def fit_normalization(records, out, cfg=None):
    """Training records only. Never compute per-crop/sector RMS normalization."""
    total_s=total_q=None; count_s=0; id_s=id_q=None; id_n=0
    for r in records:
        with np.load(cache_path(out,r['uid']),allow_pickle=False) as z:
            if cfg is not None and str(z['feature_key'].item()) != feature_key(r,cfg):
                raise ValueError('Stale cache before fitting normalization; rerun features.')
            x=z['ssl'].astype(np.float64)
            if total_s is None:total_s=np.zeros(x.shape[0]);total_q=total_s.copy()
            total_s+=x.sum(1);total_q+=(x*x).sum(1);count_s+=x.shape[1]
            a=z['id_views'].astype(np.float64);mask=z['id_masks'][:,:,None,:]
            if id_s is None:id_s=np.zeros(a.shape[2]);id_q=id_s.copy()
            id_s+=(a*mask).sum(axis=(0,1,3));id_q+=(a*a*mask).sum(axis=(0,1,3));id_n+=mask.sum()
    def stats(s,q,n):
        if n <= 0: raise ValueError('No valid training frames for normalization.')
        mean=s/n;std=np.sqrt(np.maximum(q/n-mean*mean,1.0))
        return dict(mean=mean.tolist(),std=std.tolist())
    return dict(ssl=stats(total_s,total_q,count_s),id=stats(id_s,id_q,id_n),
                fitted_recording_ids=[r['uid'] for r in records],train_only=True)
