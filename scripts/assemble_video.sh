#!/usr/bin/env bash
# assemble_video.sh — assemble la vidéo MediBoussole 3 min depuis :
#   - 20 PNG dans assets/images/scene-NN.png (générées par scripts/generate_images.py)
#   - voix-off WAV/MP3 dans assets/audio/voice-over.wav
#   - musique optionnelle dans assets/audio/music.mp3
#   - sous-titres dans assets/srt/subtitles-fr.srt
#
# Sortie : assets/final/mediboussole-3min.mp4 (1080p H.264 AAC)
#
# Prérequis : ffmpeg (brew install ffmpeg)
#
# Mode tout-IA (sans plans tournés) : marche tel quel, lance le script.
# Mode hybride : remplacez quelques scene-NN.png par scene-NN.mp4 — le script s'adapte.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMG_DIR="assets/images"
AUDIO_DIR="assets/audio"
SRT_DIR="assets/srt"
FINAL_DIR="assets/final"
mkdir -p "$FINAL_DIR"

# Durées par scène (alignées avec docs/video-storyboard.md, en secondes)
declare -a DURATIONS=(8 6 7 7 8 9 8 7 8 10 10 10 10 10 10 7 10 13 12 10)

# Vérifier qu'on a bien 20 images
missing=0
for i in $(seq 1 20); do
  num=$(printf "%02d" $i)
  if [ ! -f "$IMG_DIR/scene-$num.png" ] && [ ! -f "$IMG_DIR/scene-$num.mp4" ]; then
    echo "❌ manquant: $IMG_DIR/scene-$num.{png,mp4}"
    missing=$((missing + 1))
  fi
done
if [ $missing -gt 0 ]; then
  echo ""
  echo "⚠️  $missing scènes manquantes. Lancez :"
  echo "    python3 scripts/generate_images.py"
  echo "    (avec GOOGLE_API_KEY exportée)"
  exit 1
fi

# Vérifier voix-off
if [ ! -f "$AUDIO_DIR/voice-over.wav" ] && [ ! -f "$AUDIO_DIR/voice-over.mp3" ]; then
  echo "⚠️  Pas de voix-off dans $AUDIO_DIR/voice-over.{wav,mp3}"
  echo "    Enregistrer avec Audacity, ou ElevenLabs / Gemini TTS depuis"
  echo "    le script de docs/video-storyboard.md (champs Voix-off)."
  echo ""
  read -p "Continuer SANS voix-off (vidéo silencieuse) ? [o/N] " ans
  [[ "$ans" != "o" && "$ans" != "O" ]] && exit 1
  HAS_VOICE=0
else
  HAS_VOICE=1
fi

# Étape 1 : générer un clip de durée fixe pour chaque PNG
echo "▶ Étape 1/3 : conversion PNG → clips MP4 (Ken Burns subtil)"
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

for i in $(seq 1 20); do
  num=$(printf "%02d" $i)
  duration=${DURATIONS[$((i-1))]}
  src_png="$IMG_DIR/scene-$num.png"
  src_mp4="$IMG_DIR/scene-$num.mp4"
  out="$TMP_DIR/clip-$num.mp4"

  if [ -f "$src_mp4" ]; then
    # plan tourné — on l'utilise tel quel, recadré à 1920x1080 et trim à la durée
    ffmpeg -y -loglevel error -i "$src_mp4" -t "$duration" \
      -vf "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1" \
      -r 30 -c:v libx264 -preset fast -crf 20 -an "$out"
  else
    # PNG fixe avec léger zoom (Ken Burns 100% → 105%)
    fps=30
    total_frames=$((duration * fps))
    ffmpeg -y -loglevel error -loop 1 -i "$src_png" -t "$duration" \
      -vf "scale=2000:-1,zoompan=z='min(zoom+0.0006,1.05)':d=$total_frames:s=1920x1080:fps=$fps" \
      -r $fps -c:v libx264 -preset fast -crf 20 -an "$out"
  fi
  echo "   ✓ scène $num ($duration s)"
done

# Étape 2 : concaténer
echo "▶ Étape 2/3 : concaténation"
list="$TMP_DIR/list.txt"
> "$list"
for i in $(seq 1 20); do
  num=$(printf "%02d" $i)
  echo "file 'clip-$num.mp4'" >> "$list"
done

VIDEO_NOAUDIO="$TMP_DIR/video-noaudio.mp4"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$list" -c copy "$VIDEO_NOAUDIO"

# Étape 3 : audio + sous-titres
echo "▶ Étape 3/3 : audio + sous-titres"
OUT="$FINAL_DIR/mediboussole-3min.mp4"

VOICE_FILE=""
if [ -f "$AUDIO_DIR/voice-over.wav" ]; then
  VOICE_FILE="$AUDIO_DIR/voice-over.wav"
elif [ -f "$AUDIO_DIR/voice-over.mp3" ]; then
  VOICE_FILE="$AUDIO_DIR/voice-over.mp3"
fi

MUSIC_FILE=""
[ -f "$AUDIO_DIR/music.mp3" ] && MUSIC_FILE="$AUDIO_DIR/music.mp3"

# Construire la commande ffmpeg selon ce qu'on a
SUB_FILTER=""
if [ -f "$SRT_DIR/subtitles-fr.srt" ]; then
  SUB_FILTER=",subtitles=$SRT_DIR/subtitles-fr.srt:force_style='Fontsize=22,PrimaryColour=&Hffffff&,OutlineColour=&H80000000&,BorderStyle=3'"
fi

if [ -n "$VOICE_FILE" ] && [ -n "$MUSIC_FILE" ]; then
  # Voix + musique (musique baissée pendant la voix)
  ffmpeg -y -loglevel error \
    -i "$VIDEO_NOAUDIO" \
    -i "$VOICE_FILE" \
    -i "$MUSIC_FILE" \
    -filter_complex "[1:a]volume=1.0[v];[2:a]volume=0.15[m];[v][m]amix=inputs=2:duration=first[a]" \
    -filter:v "format=yuv420p$SUB_FILTER" \
    -map 0:v -map "[a]" \
    -c:v libx264 -preset medium -crf 20 \
    -c:a aac -b:a 192k -ar 48000 \
    -shortest "$OUT"
elif [ -n "$VOICE_FILE" ]; then
  ffmpeg -y -loglevel error \
    -i "$VIDEO_NOAUDIO" -i "$VOICE_FILE" \
    -filter:v "format=yuv420p$SUB_FILTER" \
    -map 0:v -map 1:a \
    -c:v libx264 -preset medium -crf 20 \
    -c:a aac -b:a 192k \
    -shortest "$OUT"
else
  # Vidéo silencieuse
  ffmpeg -y -loglevel error \
    -i "$VIDEO_NOAUDIO" \
    -filter:v "format=yuv420p$SUB_FILTER" \
    -c:v libx264 -preset medium -crf 20 \
    -an "$OUT"
fi

echo ""
echo "✅ Vidéo générée : $OUT"
echo ""
echo "Vérifications :"
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUT"
echo ""
echo "Lecture rapide : ffplay $OUT"
echo "Upload YouTube : https://studio.youtube.com → Upload → choisir le fichier"
