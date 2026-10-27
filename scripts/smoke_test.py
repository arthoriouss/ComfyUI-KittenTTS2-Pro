#!/usr/bin/env python3
"""Optional real-model hardware smoke test (downloads weights if not cached).

Run *outside* ComfyUI, in the same Python environment, with enough GPU/CPU memory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['auto','cuda','cpu'], default='cuda')
    ap.add_argument('--weights', choices=['packed','emb4'], default='packed')
    ap.add_argument('--voice', default='French')
    ap.add_argument('--text', default="Bonjour ! Ceci est un test de Kitten T T S deux.")
    ap.add_argument('--reference', type=Path, help='Optional WAV with 5–30 seconds of authorized single-speaker speech')
    ap.add_argument('--reference-text', default='', help='Exact transcript of the WAV (avoids Whisper pass)')
    ap.add_argument('--output', type=Path, default=Path('kitten-smoke.wav'))
    args = ap.parse_args()
    from kittenml import KittenTTS
    import soundfile as sf
    started=time.perf_counter()
    kw={'weights':args.weights}
    if args.device!='auto':
        kw['device']=args.device
    model=KittenTTS('KittenML/kitten-tts-2',**kw)
    load_time=time.perf_counter()-started
    synth_started=time.perf_counter()
    if args.reference:
        if not args.reference.is_file():
            ap.error('Reference WAV not found')
        genkw={'reference':str(args.reference)}
        if args.reference_text.strip():
            genkw['reference_text']=args.reference_text
    else:
        genkw={'voice':args.voice}
    audio=model.generate(args.text,normalize=False,**genkw)
    gen_time=time.perf_counter()-synth_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output),audio,model.sample_rate)
    duration=len(audio)/model.sample_rate
    print(f'KittenTTS2 ready: {args.output} | load={load_time:.2f}s | audio={duration:.2f}s | generation={gen_time:.2f}s | RTF={gen_time/max(duration,.001):.2f}')


if __name__ == '__main__':
    main()
