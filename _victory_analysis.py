"""One-shot spectral analysis of good_victory.wav and evil_victory.wav.

Goal: find (a) firework onset times — the discrete pops/crackles, and
(b) the boundary where fireworks hand off to the trumpet (good) or evil
laugh (evil). Mirrors the lightning HF-flux approach from effects.py +
old commits, plus a couple of complementary detectors so we can pick.

Run: uv run python _victory_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SOUNDS = Path(__file__).parent / "sounds"


def analyze(path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"FILE: {path.name}")
    print(f"{'=' * 70}")

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration:.3f}s   sr={sr}   samples={len(y)}")

    hop = 512

    # ─── 1. Standard onset_strength (broadband spectral flux) ─────────────
    env_std = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    peaks_std = librosa.util.peak_pick(
        env_std, pre_max=10, post_max=10, pre_avg=20, post_avg=20, delta=0.2, wait=10
    )
    t_std = librosa.frames_to_time(peaks_std, sr=sr, hop_length=hop)
    s_std = env_std[peaks_std]
    if len(s_std):
        s_std = s_std / s_std.max()

    # ─── 2. HF-emphasized spectral flux (transients = HF burst) ────────────
    # Same approach the lightning effect ended up using ("HF Edge"). We
    # build a magnitude spectrogram, weight high frequencies, and run
    # onset_strength on that.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # linear weight: 0 at DC, 1 at Nyquist
    hf_weight = (freqs / freqs[-1]).reshape(-1, 1)
    S_hf = S * hf_weight
    env_hf = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_hf, ref=np.max), sr=sr, hop_length=hop)
    peaks_hf = librosa.util.peak_pick(
        env_hf, pre_max=10, post_max=10, pre_avg=20, post_avg=20, delta=0.2, wait=10
    )
    t_hf = librosa.frames_to_time(peaks_hf, sr=sr, hop_length=hop)
    s_hf = env_hf[peaks_hf]
    if len(s_hf):
        s_hf = s_hf / s_hf.max()

    # ─── 3. Peak waveform amplitude in 50ms windows ────────────────────────
    win = int(0.050 * sr)
    if win > 0:
        # rolling max-abs envelope, downsampled
        n_chunks = len(y) // (win // 4)
        amp = np.array([
            np.max(np.abs(y[i * (win // 4): i * (win // 4) + win]))
            for i in range(n_chunks)
        ])
        amp = amp / (amp.max() + 1e-9)
        # peak_pick on this amp envelope
        amp_peaks = librosa.util.peak_pick(
            amp, pre_max=4, post_max=4, pre_avg=8, post_avg=8, delta=0.15, wait=4
        )
        t_amp = amp_peaks * (win / 4) / sr
        s_amp = amp[amp_peaks]
    else:
        t_amp, s_amp = np.array([]), np.array([])

    # ─── 4. RMS envelope — used to spot the firework→tail boundary ─────────
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_n = rms / (rms.max() + 1e-9)
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # ─── 5. Spectral centroid — trumpet/laugh = sustained tonal, fireworks = noisy/HF ─
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    cent_t = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop)

    # ─── 6. Spectral flatness — high during noisy fireworks, low during tonal trumpet/voice ─
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]

    # ─── Print results ────────────────────────────────────────────────────
    def _print_peaks(name, ts, ss, threshold=0.10):
        kept = [(float(t), float(s)) for t, s in zip(ts, ss) if s > threshold]
        print(f"\n  [{name}]  {len(kept)} peaks above {threshold:.2f}:")
        for t, s in kept:
            bar = "█" * int(s * 30)
            print(f"    t={t:6.3f}s  s={s:.2f}  {bar}")

    _print_peaks("standard onset_strength", t_std, s_std)
    _print_peaks("HF-flux onset_strength (lightning method)", t_hf, s_hf)
    _print_peaks("peak waveform amplitude (50ms window)", t_amp, s_amp)

    # RMS sparse trace, every ~150ms
    print("\n  [RMS envelope, ~every 150ms]")
    step = max(1, int(0.150 / (hop / sr)))
    for i in range(0, len(rms_n), step):
        bar = "█" * int(rms_n[i] * 40)
        print(f"    t={rms_t[i]:6.3f}s  rms={rms_n[i]:.2f}  {bar}")

    # Look for sustained tail — find earliest t after which rms stays > 0.2
    # for at least 0.5s without dipping below 0.05 (i.e. continuous sound)
    print("\n  [Tail-onset heuristic: where does sustained sound begin?]")
    sustained = (rms_n > 0.15).astype(int)
    win_frames = int(0.5 / (hop / sr))
    candidates = []
    for i in range(len(sustained) - win_frames):
        window = sustained[i:i + win_frames]
        if window.mean() > 0.85:  # 85% of frames are above threshold
            candidates.append(rms_t[i])
    if candidates:
        # take the first long sustained region
        first = candidates[0]
        # but require it to extend toward end of file (sustained tail)
        # — skip short clusters and take the one that reaches near end
        for t0 in candidates:
            tail_idx_start = int(t0 / (hop / sr))
            tail = sustained[tail_idx_start:]
            if tail.mean() > 0.6 and len(tail) > win_frames * 2:
                print(f"    earliest sustained block (>0.5s @ rms>0.15): t={t0:.3f}s")
                break
        else:
            print(f"    earliest sustained block: t={first:.3f}s (no long tail found)")
    else:
        print("    (no sustained block found)")

    # Spectral centroid trend — average in 0.5s windows
    print("\n  [Spectral centroid, ~every 250ms — fireworks bright, trumpet/voice lower]")
    cent_n = cent / (cent.max() + 1e-9)
    step = max(1, int(0.250 / (hop / sr)))
    for i in range(0, len(cent_n), step):
        bar = "█" * int(cent_n[i] * 30)
        print(f"    t={cent_t[i]:6.3f}s  cent={cent[i]:6.0f}Hz  {bar}")

    # Flatness trend
    flat_n = flat / (flat.max() + 1e-9)
    print("\n  [Spectral flatness, ~every 250ms — fireworks high, tonal sound low]")
    for i in range(0, len(flat_n), step):
        bar = "█" * int(flat_n[i] * 30)
        print(f"    t={cent_t[i]:6.3f}s  flat={flat[i]:.3f}  {bar}")


for name in ("good_victory.wav", "evil_victory.wav"):
    analyze(SOUNDS / name)
