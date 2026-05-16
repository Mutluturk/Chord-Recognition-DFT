"""
run_me.py  —  Chord Recognition via DFT
Signals and Systems 4CA20, 2025-2026

Generates all figures used in the analysis:
  Figure 1 — Time-domain waveform + DFT spectrum of a C major chord
  Figure 2 — Chord recognition on four chords (C maj, G maj, A min, D maj)
  Figure 3 — Parameter sensitivity: effect of window length N on Δf
  Figure 4 — Failure / limitation case: DFT cannot resolve two close frequencies

Run:  python run_me.py
Output: PNG files saved in ./figures/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy.io import wavfile

#output folder 
os.makedirs("figures", exist_ok=True)

#style 
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        11,
})


# MUSIC THEORY 
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Expected .wav filenames inside the ./recordings/ folder
WAV_FILENAMES = {
    "C major": "c_major.wav",
    "G major": "g_major.wav",
    "A minor": "a_minor.wav",
    "D major": "d_major.wav",
    "E minor": "e_minor.wav",
    "F major": "f_major.wav",
}
WAV_DIR = "recordings"

def load_signal(chord_name: str, midi_notes: list, duration: float = 1.0):
    """
    Load a real .wav recording if available in ./recordings/, otherwise
    fall back to a synthesized sine-wave chord. 
    """
    path = os.path.join(WAV_DIR, WAV_FILENAMES.get(chord_name, ""))
    if os.path.isfile(path):
        fs, data = wavfile.read(path)
        # convert to float in [-1, 1]
        if data.dtype.kind == "i":
            data = data.astype(float) / np.iinfo(data.dtype).max
        else:
            data = data.astype(float)
        # stereo → mono
        if data.ndim == 2:
            data = data.mean(axis=1)
        # trim to `duration` seconds (or pad with zeros if shorter)
        target_len = int(fs * duration)
        if len(data) >= target_len:
            data = data[:target_len]
        else:
            data = np.pad(data, (0, target_len - len(data)))
        print(f"  [RECORDED]    {chord_name} loaded from {path}")
        return data, fs, "recorded"
    else:
        sig = synthesize_chord(midi_notes, duration=duration, fs=FS)
        print(f"  [SYNTHESIZED] {chord_name}  (no .wav found at {path})")
        return sig, FS, "synthesized"

def midi_to_freq(midi: int) -> float:
    """MIDI note number → frequency in Hz.  A4 = MIDI 69 = 440 Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)

def freq_to_note(f: float) -> str:
    """Nearest note name for a frequency (e.g. 261.6 → 'C4')."""
    if f <= 0:
        return "?"
    midi = round(12 * np.log2(f / 440.0) + 69)
    octave = (midi // 12) - 1
    return NOTE_NAMES[midi % 12] + str(octave)

# Chord templates
CHORD_TEMPLATES = {
    "C major": {0, 4, 7},
    "G major": {7, 11, 2},
    "A minor": {9, 0, 4},
    "D major": {2, 6, 9},
    "E minor": {4, 7, 11},
    "F major": {5, 9, 0},
}

def chord_from_notes(detected_notes: list[str]) -> str:
    """Match a list of note names to the closest chord template."""
    pitch_classes = set()
    for n in detected_notes:
        for i, name in enumerate(NOTE_NAMES):
            if n.startswith(name):
                pitch_classes.add(i)
                break
    best, best_score = "Unknown", 0
    for chord, template in CHORD_TEMPLATES.items():
        score = len(pitch_classes & template) / max(len(template), 1)
        if score > best_score:
            best, best_score = chord, score
    return best if best_score >= 0.6 else "Unknown"

# SIGNAL SYNTHESIS
FS = 44100  # sample rate (Hz)

def synthesize_chord(midi_notes: list[int], duration: float = 0.5,
                     fs: int = FS, amplitude: float = 1.0) -> np.ndarray:
    """Sum of sine waves, one per MIDI note. Equal amplitude, Hann-windowed."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)
    for m in midi_notes:
        signal += np.sin(2 * np.pi * midi_to_freq(m) * t)
    signal /= len(midi_notes)            # normalise
    window = np.hanning(len(signal))
    return signal * window * amplitude

def compute_dft(signal: np.ndarray, fs: int = FS):
    """Return one-sided frequency axis and magnitude spectrum."""
    N = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(spectrum) / N
    return freqs, magnitude

def find_peaks(freqs, magnitude, threshold_ratio: float = 0.1,
               f_min: float = 80, f_max: float = 1400) -> list[float]:
    """Return frequencies of spectral peaks above threshold in [f_min, f_max]."""
    from scipy.signal import find_peaks as sp_peaks
    mask = (freqs >= f_min) & (freqs <= f_max)
    mag_masked = magnitude.copy()
    mag_masked[~mask] = 0
    threshold = threshold_ratio * mag_masked.max()
    peak_idx, _ = sp_peaks(mag_masked, height=threshold, distance=20)
    return sorted(freqs[peak_idx].tolist())

# FIGURE 1 — Time domain + DFT spectrum of C major
def figure1_basic_dft():
    """Show time-domain signal and its DFT for a C major chord."""
    C_MAJOR = [60, 64, 67]   # C4, E4, G4
    freqs_true = [midi_to_freq(m) for m in C_MAJOR]

    signal, fs, source = load_signal("C major", C_MAJOR, duration=0.5)
    freqs, mag = compute_dft(signal, fs=fs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Figure 1 — C major chord: time domain and DFT spectrum", fontweight="bold")

    # time domain (show first 10 ms)
    t = np.arange(len(signal)) / FS * 1000
    ax = axes[0]
    ax.plot(t[:int(0.01 * FS)], signal[:int(0.01 * FS)], color="#3B5FC0", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time-domain waveform (first 10 ms)")

    # DFT spectrum
    ax = axes[1]
    ax.plot(freqs, mag, color="#3B5FC0", lw=1)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"DFT magnitude spectrum  (N = {len(signal)},  Δf = {fs/len(signal):.2f} Hz,  source: {source})")

    note_labels = ["C4 (261.6 Hz)", "E4 (329.6 Hz)", "G4 (392.0 Hz)"]
    colors = ["#E8593C", "#2A9D8F", "#E9C46A"]
    for f, label, c in zip(freqs_true, note_labels, colors):
        ax.axvline(f, color=c, linestyle="--", lw=1.2, label=label)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig1_basic_dft.png", dpi=150)
    plt.close()
    print("Saved: figures/fig1_basic_dft.png")


# FIGURE 2 — Chord recognition on four chords

def figure2_chord_recognition():
    """Detect and label four different chords from their DFT spectra."""
    chords = {
        "C major": [60, 64, 67],
        "G major": [67, 71, 74],
        "A minor": [69, 72, 76],
        "D major": [62, 66, 69],
    }
    colors = ["#3B5FC0", "#2A9D8F", "#E8593C", "#8B5CF6"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Figure 2 — Chord recognition: DFT peak detection", fontweight="bold")
    axes = axes.flatten()

    for ax, (name, notes), color in zip(axes, chords.items(), colors):
        signal, fs, source = load_signal(name, notes)
        freqs, mag = compute_dft(signal, fs=fs)
        detected_freqs = find_peaks(freqs, mag)
        detected_notes = [freq_to_note(f) for f in detected_freqs]
        predicted = chord_from_notes(detected_notes)

        ax.plot(freqs, mag, color=color, lw=1)
        ax.set_xlim(0, 1000)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Magnitude", fontsize=9)
        ax.set_title(f"Input: {name}  [{source}]   →   Detected: {predicted}", fontsize=10)

        for f, note in zip(detected_freqs, detected_notes):
            ax.annotate(note, xy=(f, mag[np.argmin(np.abs(freqs - f))]),
                        xytext=(f + 10, mag[np.argmin(np.abs(freqs - f))] * 1.05),
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

    plt.tight_layout()
    plt.savefig("figures/fig2_chord_recognition.png", dpi=150)
    plt.close()
    print("Saved: figures/fig2_chord_recognition.png")

# FIGURE 3 — Parameter sensitivity: window length N


def figure3_parameter_sensitivity():
    """
    Stress test A: Vary window length N.
    Shorter N → coarser Δf = fs/N → peaks blur together.
    """
    C_MAJOR = [60, 64, 67]
    window_lengths = [256, 512, 1024, 8192]
    freqs_true = [midi_to_freq(m) for m in C_MAJOR]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Figure 3 — Parameter sensitivity: effect of window length N on Δf",
                 fontweight="bold")
    axes = axes.flatten()

    full_signal, fs, source = load_signal("C major", C_MAJOR, duration=1.0)

    for ax, N in zip(axes, window_lengths):
        segment = full_signal[:N]
        freqs, mag = compute_dft(segment, fs=fs)
        delta_f = fs / N

        ax.plot(freqs, mag, color="#3B5FC0", lw=1)
        ax.set_xlim(200, 500)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Magnitude", fontsize=9)
        ax.set_title(f"N = {N}   →   Δf = {delta_f:.1f} Hz", fontsize=10)

        colors = ["#E8593C", "#2A9D8F", "#E9C46A"]
        for f, c in zip(freqs_true, colors):
            ax.axvline(f, color=c, linestyle="--", lw=1, alpha=0.7)

    # shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#E8593C", linestyle="--", label="C4 (261.6 Hz)"),
        Line2D([0], [0], color="#2A9D8F", linestyle="--", label="E4 (329.6 Hz)"),
        Line2D([0], [0], color="#E9C46A", linestyle="--", label="G4 (392.0 Hz)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig("figures/fig3_parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/fig3_parameter_sensitivity.png")


# FIGURE 4 — Failure / limitation case: unresolvable close frequencies


def figure4_failure_case():
    """
    Stress test B: Detune E4 by a small amount Δ.
    When Δ < Δf = fs/N, the DFT merges both into one peak — failure!
    Shows three detuning amounts at two window sizes.
    """
    E4 = midi_to_freq(64)          # 329.63 Hz
    detunings = [0, 5, 15]         # Hz offset applied to second 'string'
    N_small, N_large = 2048, 32768

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle("Figure 4 — Failure case: DFT cannot resolve closely spaced frequencies",
                 fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def spectrum_of_two_tones(f1, f2, N):
        t = np.arange(N) / FS
        sig = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
        window = np.hanning(N)
        spec = np.abs(np.fft.rfft(sig * window)) / N
        freqs = np.fft.rfftfreq(N, d=1/FS)
        return freqs, spec

    row_labels = [f"N = {N_small}  (Δf = {FS/N_small:.1f} Hz)",
                  f"N = {N_large}  (Δf = {FS/N_large:.2f} Hz)"]

    for row, N in enumerate([N_small, N_large]):
        delta_f = FS / N
        for col, detune in enumerate(detunings):
            ax = fig.add_subplot(gs[row, col])
            f1, f2 = E4, E4 + detune
            freqs, mag = spectrum_of_two_tones(f1, f2, N)

            can_resolve = detune >= delta_f
            status = "✓ resolved" if (detune > 0 and can_resolve) else \
                     ("✗ NOT resolved" if detune > 0 else "reference")
            color = "#2A9D8F" if can_resolve else "#E8593C"

            ax.plot(freqs, mag, color=color, lw=1)
            ax.set_xlim(E4 - 50, E4 + 80)
            ax.set_xlabel("Frequency (Hz)", fontsize=8)
            ax.set_ylabel("Magnitude", fontsize=8)
            title = f"Δ = {detune} Hz   {status}" if detune > 0 else f"Δ = 0 Hz  (no detune)"
            ax.set_title(title, fontsize=9, color=color if detune > 0 else "black")

            ax.axvline(f1, color="gray", linestyle=":", lw=1, alpha=0.6)
            if detune > 0:
                ax.axvline(f2, color="gray", linestyle=":", lw=1, alpha=0.6)

            if col == 0:
                ax.set_ylabel(row_labels[row] + "\n\nMagnitude", fontsize=8)

    plt.savefig("figures/fig4_failure_case.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/fig4_failure_case.png")


# FIGURE 5 — Summary: detection accuracy vs window length

def figure5_accuracy_vs_N():
    """
    Quantitative summary: fraction of chords correctly identified
    as a function of window length N. Shows where the DFT becomes reliable.
    """
    test_chords = {
        "C major": [60, 64, 67],
        "G major": [67, 71, 74],
        "A minor": [69, 72, 76],
        "D major": [62, 66, 69],
        "E minor": [64, 67, 71],
        "F major": [65, 69, 72],
    }
    window_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    accuracy = []
    full_signals = {}
    sample_rates = {}
    for name, notes in test_chords.items():
        sig, fs, _ = load_signal(name, notes, duration=1.0)
        full_signals[name] = sig
        sample_rates[name] = fs

    for N in window_lengths:
        correct = 0
        for name, notes in test_chords.items():
            seg = full_signals[name][:N]
            fs = sample_rates[name]
            freqs, mag = compute_dft(seg, fs=fs)
            peaks = find_peaks(freqs, mag)
            detected = [freq_to_note(f) for f in peaks]
            predicted = chord_from_notes(detected)
            if predicted == name:
                correct += 1
        accuracy.append(correct / len(test_chords))

    delta_fs = [FS / N for N in window_lengths]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Figure 5 — Detection accuracy vs. window length N", fontweight="bold")

    ax.plot(window_lengths, [a * 100 for a in accuracy],
            marker="o", color="#3B5FC0", lw=2, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Window length N (samples, log scale)")
    ax.set_ylabel("Recognition accuracy (%)")
    ax.set_ylim(-5, 110)
    ax.set_xticks(window_lengths)
    ax.set_xticklabels([f"{N}\n(Δf≈{FS/N:.0f}Hz)" for N in window_lengths], fontsize=8)

    ax.axhline(100, color="#2A9D8F", linestyle="--", lw=1, label="100% accuracy")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig5_accuracy_vs_N.png", dpi=150)
    plt.close()
    print("Saved: figures/fig5_accuracy_vs_N.png")


# MAIN
if __name__ == "__main__":
    print("=== Chord Recognition via DFT — generating all figures ===\n")
    recorded = [name for name, fname in WAV_FILENAMES.items()
                if os.path.isfile(os.path.join(WAV_DIR, fname))]
    if recorded:
        print(f"Found recordings for: {', '.join(recorded)}")
    else:
        print(f"No .wav files found in ./{WAV_DIR}/ — using synthesized signals.")
        print(f"To use real recordings, place files like 'c_major.wav' in ./{WAV_DIR}/\n")
    figure1_basic_dft()
    figure2_chord_recognition()
    figure3_parameter_sensitivity()
    figure4_failure_case()
    figure5_accuracy_vs_N()
