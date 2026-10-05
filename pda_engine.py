#!/usr/bin/env python3
# pda voice: edge tts -> suit/underwater comm fx -> play
#
# colab:
#   st = await speak("Welcome aboard, Captain. *Databank* online.")
#   from IPython.display import Audio, display; display(Audio(st.T, rate=FS))
# cli:
#   python3 pda_engine.py "Welcome aboard" [--save out.wav]
# *stars* around a word = stress it (only if STRESS is on)
import argparse
import asyncio
import io
import functools
import random
import re
import shutil
import subprocess
import sys
import wave

import edge_tts
import numpy as np
from pydub import AudioSegment
from scipy.signal import (butter, fftconvolve, istft, lfilter,
                          resample_poly, sosfilt, stft)

VOICE = "en-GB-SoniaNeural"
FS = 44100

# metallic
ROBOT_MIX = 0        # 0 natural .. 1 full vocoder buzz
ROBOT_HOP = 220      # buzz pitch = FS/hop
METAL_HZ = 1800
METAL_DB = 3         # 0 = off
METAL_Q = 3.0
FLANGE_MIN_MS = 0.6
FLANGE_MAX_MS = 3.0
FLANGE_MIX = 0.15    # 0 = off

# suit
BAND = (250, 7500)   # speaker band, lower the top for more muffle
WOBBLE_HZ = 1.1
WOBBLE_DEPTH = 32    # samples, 0 = off
REVERB_MS = 110
REVERB_MIX = 0.3
HISS = 0
RUMBLE = 0.03
HUM_GAIN = 0.5
TAIL_S = 0.5

# water
CHORUS_DEPTH = 220   # samples
CHORUS_MIX = 0.35    # .35 light, .5 watery, .7 very
SWEEP_LO = 3500
SWEEP_HI = 9000
SWEEP_HZ = 0.35
BUBBLES = 1.2        # pops/sec, 0 = off
BUBBLE_GAIN = 0.04

# delivery
STRESS = False       # False = flat, ignores *stars* and moods
PITCH_FLATTEN = 0.85 # 0 natural .. 1 monotone, needs praat-parselmouth

# beep
BEEP = True
BEEP_HZ = (740, 1110)
BEEP_GAIN = 0.45
BEEP_GAP = 0.12

# moods: (rate %, pitch Hz, volume %)
MOODS = {
    "neutral": (0, 0, 0),
    "alert": (0, 0, 5),
    "warm": (-3, 0, -3),
}
EMPH_OFFSET = (-6, 4, 8)
CLAUSE_SPLIT = False  # one clip per clause, you can hear the seams
JITTER = (0.0, 0.0)   # only matters with CLAUSE_SPLIT
OPENER_LIFT = 0
QUESTION_LIFT = 0
FINAL_DROP = (0, 0)
PIECE_GAP = 0.03
PAUSE = {",": 0.06, ";": 0.08, ":": 0.08, ".": 0.12, "!": 0.12, "?": 0.14, "…": 0.2}
DASH_PAUSE = 0.06
ALERT_RE = re.compile(r"\b(hot|low|behind|piling|tight|at max|real load|plug in)\b", re.I)
WARM_RE = re.compile(r"\b(wish|thinking of you|your voice|reminds me|you'd love|"
                     r"next to me|around|playlist for you|better with you)\b", re.I)
# auto stress if no stars: numbers + a few words
AUTO_RE = re.compile(r"(\d+(?:[.,]\d+)?(?:°C|%|h\b)?"
                     r"|\b(?:hot|cool|low|tight|behind|piling|midnight|late|early|straight)\b)", re.I)


# ---- speech ----

def trim(x, thresh=0.01, keep=0.01):
    # tts pads silence on both ends
    idx = np.flatnonzero(np.abs(x) > thresh)
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(FS * keep))
    b = min(len(x), idx[-1] + int(FS * keep))
    return x[a:b]


async def _tts(text, voice, **kw):
    buf = bytearray()
    async for ch in edge_tts.Communicate(text, voice, **kw).stream():
        if ch["type"] == "audio":
            buf += ch["data"]
    if not buf:
        raise RuntimeError(f"no audio for {text!r}")
    seg = AudioSegment.from_file(io.BytesIO(bytes(buf)), format="mp3")
    seg = seg.set_channels(1).set_frame_rate(FS)
    return trim(np.array(seg.get_array_of_samples(), dtype=np.float64) / 32768.0)


def _kw(rate, pitch, vol):
    return dict(rate=f"{round(rate):+d}%", pitch=f"{round(pitch):+d}Hz",
                volume=f"{round(vol):+d}%")


def plan(text):
    # -> [(pieces, pause_after)], pieces = [(text, tts kwargs)]
    if not STRESS:
        text = text.replace("*", "")
        mood = "neutral"
    else:
        if "*" not in text:
            text = AUTO_RE.sub(r"*\1*", text)
        mood = "alert" if ALERT_RE.search(text) else "warm" if WARM_RE.search(text) else "neutral"
    base_r, base_p, base_v = MOODS[mood]

    if CLAUSE_SPLIT:
        clauses = [c for c in re.split(r"(?<=[,.;:!?…])\s+|\s+—\s+", text) if c.strip()]
    else:
        clauses = [text]

    rng = random.Random()
    out = []
    for i, c in enumerate(clauses):
        end = c.rstrip()[-1:]
        last = i == len(clauses) - 1
        q = end == "?"
        r = base_r + rng.uniform(-JITTER[0], JITTER[0]) - (FINAL_DROP[0] if last else 0)
        p = (base_p + rng.uniform(-JITTER[1], JITTER[1])
             + (OPENER_LIFT if i == 0 else 0) + (QUESTION_LIFT if q else 0)
             - (FINAL_DROP[1] if last and not q else 0))
        pieces = []
        for part in re.split(r"(\*[^*]+\*)", c):
            if not re.search(r"\w", part):
                continue
            er, ep, ev = EMPH_OFFSET if part.startswith("*") else (0, 0, 0)
            pieces.append([part.strip("*").strip().lstrip("—– "), _kw(r + er, p + ep, base_v + ev)])
        if pieces and end in PAUSE and not pieces[-1][0].endswith(end):
            pieces[-1][0] += end  # keep the punctuation so edge falls/rises
        out.append((pieces, 0.0 if last else PAUSE.get(end, DASH_PAUSE)))
    return out


async def synth(text, voice=VOICE):
    steps = plan(text)
    flat = [pc for pieces, _ in steps for pc in pieces]
    audios = iter(await asyncio.gather(*[_tts(t, voice, **kw) for t, kw in flat]))
    chunks = []
    for pieces, pause in steps:
        for _ in pieces:
            chunks += [next(audios), np.zeros(int(FS * PIECE_GAP))]
        if pause:
            chunks.append(np.zeros(int(FS * pause)))
    return np.concatenate(chunks)


# ---- fx ----

def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def comb(x, delay, g):
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -g
    return lfilter([1.0], a, x)


def robotize(x):
    nper, nover = 1024, 1024 - ROBOT_HOP
    _, _, z = stft(x, FS, nperseg=nper, noverlap=nover)
    _, y = istft(np.abs(z), FS, nperseg=nper, noverlap=nover)  # drop phase -> buzz
    y = np.pad(y[:len(x)], (0, max(0, len(x) - len(y))))
    return y * (_rms(x) / max(_rms(y), 1e-9))


def peaking_sos(f0, gain_db, q):
    # RBJ cookbook peaking eq
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / FS
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return (np.array(b + a) / a[0])[None, :]


def flange(x, rate_hz, phase):
    t = np.arange(len(x), dtype=np.float64)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t / FS + phase)
    d = (FLANGE_MIN_MS + (FLANGE_MAX_MS - FLANGE_MIN_MS) * lfo) * FS / 1000
    return x + FLANGE_MIX * np.interp(t - d, t, x)


def wobble(x, phase):
    t = np.arange(len(x), dtype=np.float64)
    d = WOBBLE_DEPTH * np.sin(2 * np.pi * WOBBLE_HZ * t / FS + phase)
    return np.interp(t - d, t, x)


def water(x, phase):
    # 3 delay wobbles at different speeds
    t = np.arange(len(x), dtype=np.float64)
    out = x * (1 - CHORUS_MIX)
    for hz, depth, ph in ((0.9, 1.0, 0.0), (2.7, 0.55, 2.1), (6.3, 0.2, 4.2)):
        d = CHORUS_DEPTH * depth * (1 + np.sin(2 * np.pi * hz * t / FS + phase + ph)) / 2 + 8
        out = out + (CHORUS_MIX / 3) * np.interp(t - d, t, x)
    return out


def sweep(x):
    # lowpass that drifts between SWEEP_LO and SWEEP_HI
    t = np.arange(len(x))
    lo = sosfilt(butter(2, SWEEP_LO, fs=FS, output="sos"), x)
    hi = sosfilt(butter(2, SWEEP_HI, fs=FS, output="sos"), x)
    a = 0.5 + 0.5 * np.sin(2 * np.pi * SWEEP_HZ * t / FS)
    return a * lo + (1 - a) * hi


def bubbles(n, rng):
    out = np.zeros(n)
    for _ in range(int(BUBBLES * n / FS)):
        length = int(FS * rng.uniform(0.04, 0.11))
        if length >= n:
            continue
        start = int(rng.integers(0, n - length))
        tt = np.arange(length) / FS
        dur = length / FS
        f0 = rng.uniform(350, 700)
        ph = 2 * np.pi * (f0 * tt + f0 * 1.5 * tt ** 2 / (2 * dur))  # f0 -> 2.5*f0
        env = (1 - np.exp(-tt / 0.003)) * np.exp(-tt / (dur / 3))
        out[start:start + length] += rng.uniform(0.4, 1.0) * BUBBLE_GAIN * env * np.sin(ph)
    return out


def room(x, rng):
    n = int(FS * REVERB_MS / 1000)
    t = np.arange(n) / FS
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / (REVERB_MS / 1000))
    ir = sosfilt(butter(2, 4500, fs=FS, output="sos"), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    return x + REVERB_MIX * fftconvolve(x, ir)[:len(x)]


def ambience(n, rng):
    out = np.zeros(n)
    if HISS:
        h = sosfilt(butter(2, [2000, 6000], btype="band", fs=FS, output="sos"),
                    rng.standard_normal(n))
        out += h * HISS / max(np.std(h), 1e-9)
    if RUMBLE:
        r = sosfilt(butter(2, 120, fs=FS, output="sos"), rng.standard_normal(n))
        swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.23 * np.arange(n) / FS)
        out += r * swell * RUMBLE / max(np.std(r), 1e-9)
    return out


def flatten_pitch(x, amount=None):
    amount = PITCH_FLATTEN if amount is None else amount
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        print("skipping pitch flatten, pip install praat-parselmouth", file=sys.stderr)
        return x
    snd = parselmouth.Sound(x, sampling_frequency=FS)
    manip = call(snd, "To Manipulation", 0.01, 100, 500)
    pt = call(manip, "Extract pitch tier")
    n = int(call(pt, "Get number of points"))
    if n < 2:
        return x
    times = [call(pt, "Get time from index", i) for i in range(1, n + 1)]
    freqs = np.array([call(pt, "Get value at index", i) for i in range(1, n + 1)])
    med = float(np.median(freqs))
    flat = call("Create PitchTier", "flat", snd.xmin, snd.xmax)
    for t, f in zip(times, freqs):
        call(flat, "Add point", t, med * (f / med) ** (1 - amount))
    call([flat, manip], "Replace pitch tier")
    y = np.asarray(call(manip, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)
    y = np.pad(y[:len(x)], (0, max(0, len(x) - len(y))))
    return y * (_rms(x) / max(_rms(y), 1e-9))


def pda_fx(x, stretch=False, tail=None, voice=True):
    rng = np.random.default_rng(7)  # fixed seed so it sounds the same every run

    x = resample_poly(x, 100, 97 if stretch else 100)
    if voice and PITCH_FLATTEN:
        x = flatten_pitch(x)
    x = np.pad(x, (0, int((TAIL_S if tail is None else tail) * FS)))
    x /= max(np.max(np.abs(x)), 1e-9)

    hum = sosfilt(butter(2, 200, fs=FS, output="sos"), x)  # grab the low end before filtering

    if ROBOT_MIX:
        x = (1 - ROBOT_MIX) * x + ROBOT_MIX * robotize(x)
    x = sosfilt(butter(3, BAND, btype="band", fs=FS, output="sos"), x)
    if METAL_DB:
        x = sosfilt(peaking_sos(METAL_HZ, METAL_DB, METAL_Q), x)
    if WOBBLE_DEPTH:
        x = wobble(x, 0.0)
    if SWEEP_HZ:
        x = sweep(x)

    # left/right get slightly different flanger + comb settings
    chans = []
    for flange_hz, delay_ms, g, phase in ((0.25, 3, -0.12, 0.0), (0.31, 5, -0.15, 1.7)):
        c = flange(x, flange_hz, phase) if FLANGE_MIX else x
        if CHORUS_MIX:
            c = water(c, phase)
        c = comb(c, round(FS * delay_ms / 1000), g)
        chans.append(room(c, rng))
    st = np.column_stack(chans)

    st += HUM_GAIN * hum[:, None]
    st += np.column_stack([ambience(len(x), rng), ambience(len(x), rng)])
    if BUBBLES:
        st += np.column_stack([bubbles(len(x), rng), bubbles(len(x), rng)])
    return st * (0.95 / max(np.max(np.abs(st)), 1e-9))


@functools.lru_cache(maxsize=1)
def make_beep():
    t = np.arange(int(FS * 0.2)) / FS
    f = np.where(t < 0.09, BEEP_HZ[0], BEEP_HZ[1])
    env = np.minimum(1.0, t / 0.006) * np.exp(-t / 0.08)
    x = np.sin(2 * np.pi * np.cumsum(f) / FS) * env
    return pda_fx(x, False, tail=0.1, voice=False) * BEEP_GAIN


def prepend_beep(st):
    if not BEEP:
        return st
    return np.vstack([make_beep(), np.zeros((int(FS * BEEP_GAP), 2)), st])


# ---- output ----

def to_wav_bytes(st):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def play(wav):
    for cmd in (["aplay", "-q", "-"],
                ["paplay"],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=wav, check=True)
            return
    print("no player found (aplay/paplay/ffplay), use --save", file=sys.stderr)


async def speak(text, stretch=None, voice=VOICE):
    if stretch is None:
        stretch = "databank" in text.lower()
    print("\n[PDA Spatial Core] Fetching crisp voice baseline data...")
    audio = await synth(text, voice)
    print("[PDA Spatial Core] Injecting metallic suit-comm matrix...")
    st = prepend_beep(pda_fx(audio, stretch))
    print("[PDA Spatial Core] Broadcast active...\n")
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="*")
    p.add_argument("--stretch", action="store_true", help="slower, deeper")
    p.add_argument("--voice", default=VOICE)
    p.add_argument("--save", metavar="FILE")
    p.add_argument("--no-play", action="store_true")
    args = p.parse_args()

    text = " ".join(args.text) or input("Enter text for PDA: ")
    if not text.strip():
        sys.exit(1)

    st = asyncio.run(speak(text, True if args.stretch else None, args.voice))
    wav = to_wav_bytes(st)
    if args.save:
        with open(args.save, "wb") as f:
            f.write(wav)
        print(f"[PDA Spatial Core] Saved to {args.save}")
    if not args.no_play:
        play(wav)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
