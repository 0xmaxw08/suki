# PDA — AI Desktop Companion

A desktop companion that talks to you through a metallic, underwater suit-comm voice, like an in-suit PDA. A small local LLM handles the conversation, and a custom voice engine turns its replies into speech.

> **Status:** early development. The voice engine works. The rest is planned. See the [roadmap](#roadmap).

## How it works

```
 you (text / mic)
       |
       v
 +-------------+     +--------------+     +--------------------+
 |  input      | --> |  LLM         | --> |  voice engine      | --> speakers
 |  (planned)  |     |  Qwen 2.5    |     |  pda_engine.py     |
 +-------------+     |  Coder 3B    |     |  edge TTS + FX     |
                     +--------------+     +--------------------+
                            ^
                            |
                    persona prompt + memory (planned)
```

| Part | Tech | Status |
|---|---|---|
| Voice | Edge neural TTS + custom DSP chain (`pda_engine.py`) | Working |
| Brain | Qwen 2.5 Coder 3B, run locally | Planned |
| Desktop UI | TBD | Planned |
| Memory | TBD | Planned |

## Voice engine

`pda_engine.py` takes text and returns a stereo audio array. It works in two stages:

1. **Synthesis.** Microsoft Edge neural TTS (`en-GB-SoniaNeural` by default) reads the text. Delivery is deliberately flat and corporate: pitch is pulled toward its median with Praat.
2. **Suit-comm FX.** The clean voice goes through a band-limited small-speaker filter, a resonant "tin" peak, a swept flanger, a chorus and drifting low-pass for the underwater feel, a short helmet reverb, low rumble, random bubble pops and a two-note beep before each line.

The effects use a fixed random seed, so the same line sounds the same every time.

### Install

```bash
pip install edge-tts numpy scipy pydub
pip install praat-parselmouth   # optional, needed for pitch flattening
```

You also need **ffmpeg** on your PATH (pydub uses it to decode the TTS output), and an internet connection, because Edge TTS is an online service. For playback on Linux you need one of `aplay`, `paplay` or `ffplay`.

### Use it

Command line:

```bash
python3 pda_engine.py "Welcome aboard, Captain."
python3 pda_engine.py "Databank online." --save out.wav --no-play
```

Python or Colab:

```python
from pda_engine import speak, FS

st = await speak("Welcome aboard, Captain. *Databank* online.")
# st is a (N, 2) float array at FS (44100 Hz)

from IPython.display import Audio, display
display(Audio(st.T, rate=FS))
```

Flags: `--stretch` (slower and deeper; turns on automatically if the text contains "databank"), `--voice`, `--save FILE`, `--no-play`.

### Tuning

All the knobs are constants at the top of `pda_engine.py`. The ones you will touch most:

| Constant | Effect |
|---|---|
| `VOICE` | Any Edge TTS voice name |
| `PITCH_FLATTEN` | 0 = natural intonation, 1 = dead monotone |
| `STRESS` | `True` enables `*star*` emphasis, auto-stress and moods |
| `BAND` | Speaker band in Hz; lower the top number for more muffling |
| `CHORUS_MIX` | How watery it sounds (0.35 light, 0.5 clear, 0.7 heavy) |
| `BUBBLES` | Bubble pops per second, 0 turns them off |
| `METAL_DB` / `FLANGE_MIX` | Strength of the metallic shimmer |
| `BEEP` | Beep before each line on or off |

## The AI model

The plan is to run **Qwen 2.5 Coder 3B** locally. The easiest route is [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:3b
ollama run qwen2.5-coder:3b
```

Two things to know:

- This is a code-tuned model, so it may be weaker at casual conversation and staying in character than a general chat model. A strong system prompt helps. If the personality comes out stiff, try the general instruct variant (`qwen2.5:3b`) as a drop-in swap.
- 3B models run on most modern machines, but expect slower replies on CPU-only setups. Keep replies short, since the voice has to read them out.

## Planned project layout

```
pda/
├── pda_engine.py     # voice (done)
├── brain.py          # LLM client + persona prompt
├── memory.py         # conversation history / long-term notes
├── app.py            # desktop UI + main loop
├── config.py         # model name, voice, FX presets
└── README.md
```

Nothing except `pda_engine.py` exists yet. The layout is a sketch.

## Roadmap

- [x] Voice engine with suit-comm FX
- [ ] Connect the LLM (Ollama) and write the persona prompt
- [ ] Text chat loop: type, get a spoken reply
- [ ] Desktop window (always-on-top, small, draggable)
- [ ] Speech-to-text input
- [ ] Conversation memory
- [ ] Reply streaming: start speaking sentence by sentence while the LLM is still generating
- [ ] Voice and FX presets in a config file
- [ ] Offline TTS fallback (Edge TTS needs internet)

## Known limitations

- Edge TTS needs an internet connection, so the voice is not fully offline yet.
- The FX chain renders the whole line before playback, so long replies add delay. Sentence-level streaming is on the roadmap.
- Tested only through the CLI and Colab so far.

## License

[MIT](LICENSE)
