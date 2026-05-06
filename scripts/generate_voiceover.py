"""
Génère la voix-off française MediBoussole.

Stratégie : extrait les lignes Voix-off du storyboard, génère un AIFF par scène
avec macOS `say` (voix Thomas FR), puis concatène avec des silences calibrés
sur les durées de chaque scène pour que la voix tombe bien sur les images.

Sortie : assets/audio/voice-over.wav (3 min, 48 kHz mono)

Usage :
    python scripts/generate_voiceover.py
    python scripts/generate_voiceover.py --voice Audrey --rate 180
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORYBOARD = ROOT / "docs" / "video-storyboard.md"
OUT_WAV = ROOT / "assets" / "audio" / "voice-over.wav"
TMP_DIR = ROOT / "assets" / "audio" / ".tmp_voice"

# Durées de chaque scène en secondes (cf. storyboard)
SCENE_DURATIONS = [8, 6, 7, 7, 8, 9, 8, 7, 8, 10, 10, 10, 10, 10, 10, 7, 10, 13, 12, 10]
assert sum(SCENE_DURATIONS) == 180, f"Total ≠ 180s : {sum(SCENE_DURATIONS)}"


def extract_voiceover() -> list[str]:
    """Extrait les 20 lignes Voix-off du storyboard (vide si scène texte-écran seul)."""
    text = STORYBOARD.read_text()
    # Match scene blocks
    scenes = re.split(r"### Scène \d+ — ", text)[1:]  # premier split avant scène 1
    voiceovers = []
    for sc in scenes[:20]:
        # Match Voix-off
        m = re.search(r"\*\*Voix-off\*\*\s*:\s*« ?(.+?) ?»", sc)
        if m:
            voiceovers.append(m.group(1).strip())
        else:
            # Fallback : Texte écran (lu aussi mais c'est moins idéal)
            m2 = re.search(r"\*\*Texte écran\*\*\s*:\s*« ?(.+?) ?»", sc)
            if m2:
                voiceovers.append(m2.group(1).strip())
            else:
                voiceovers.append("")  # silence
    return voiceovers


def say_to_aiff(text: str, out_aiff: Path, voice: str, rate: int) -> None:
    """macOS `say` → AIFF mono."""
    cmd = ["say", "-v", voice, "-r", str(rate), "-o", str(out_aiff), text]
    subprocess.run(cmd, check=True, capture_output=True)


def aiff_to_wav_padded(aiff: Path, target_seconds: float, out_wav: Path) -> None:
    """Convertit AIFF → WAV mono 48 kHz, paddé à target_seconds avec silence à la fin."""
    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(aiff)],
        capture_output=True, text=True
    )
    speech_dur = float(probe.stdout.strip() or 0)
    pad = max(target_seconds - speech_dur - 0.4, 0.0)  # 0.4s pause avant scène suivante

    # Convert + pad
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(aiff),
        "-af", f"apad=pad_dur={pad},aformat=channel_layouts=mono,aresample=48000",
        "-t", str(target_seconds),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def silent_wav(seconds: float, out_wav: Path) -> None:
    """WAV silence pour scènes sans voix-off."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono",
        "-t", str(seconds),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="Thomas",
                        help="Voix macOS (Thomas/Audrey/Aurelie en FR)")
    parser.add_argument("--rate", type=int, default=170,
                        help="Vitesse de parole (mots/min, défaut 170)")
    args = parser.parse_args()

    if shutil.which("say") is None:
        sys.exit("Erreur : `say` non trouvé. Disponible uniquement sur macOS.")
    if shutil.which("ffmpeg") is None:
        sys.exit("Erreur : ffmpeg requis. brew install ffmpeg")

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    voiceovers = extract_voiceover()
    print(f"Voix-off extraites : {sum(1 for v in voiceovers if v)}/{len(voiceovers)} non-vides\n")

    # Générer chaque scène
    scene_wavs = []
    for i, (text, dur) in enumerate(zip(voiceovers, SCENE_DURATIONS), 1):
        out = TMP_DIR / f"scene-{i:02d}.wav"
        if not text:
            silent_wav(dur, out)
            print(f"  [{i:02d}/20] silence ({dur}s)")
        else:
            aiff = TMP_DIR / f"scene-{i:02d}.aiff"
            try:
                say_to_aiff(text, aiff, args.voice, args.rate)
                aiff_to_wav_padded(aiff, dur, out)
                preview = text[:50] + ("…" if len(text) > 50 else "")
                print(f"  [{i:02d}/20] ✓ « {preview} »")
            except subprocess.CalledProcessError as e:
                print(f"  [{i:02d}/20] ✗ {e}")
                silent_wav(dur, out)
        scene_wavs.append(out)

    # Concaténer toutes les scènes
    print("\nConcaténation finale...")
    concat_list = TMP_DIR / "concat.txt"
    concat_list.write_text("\n".join(f"file '{w.name}'" for w in scene_wavs))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(OUT_WAV),
    ], check=True)

    # Probe final duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(OUT_WAV)],
        capture_output=True, text=True
    )
    final_dur = float(probe.stdout.strip())
    size_kb = OUT_WAV.stat().st_size / 1024

    print(f"\n✓ Voix-off : {OUT_WAV.relative_to(ROOT)}")
    print(f"  Durée : {final_dur:.1f} s (cible 180 s)")
    print(f"  Taille : {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
