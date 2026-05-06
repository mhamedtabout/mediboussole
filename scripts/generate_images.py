"""
Génère les 20 images de la vidéo MediBoussole en batch.

Deux providers supportés :
- 'pollinations' (défaut) : gratuit, sans clé API, modèle Flux
- 'imagen'                : qualité supérieure, nécessite GOOGLE_API_KEY

Lit les prompts depuis docs/video-storyboard.md et sauvegarde
dans assets/images/scene-NN.png.

Usage :
    python scripts/generate_images.py                          # pollinations par défaut
    python scripts/generate_images.py --provider imagen        # nécessite GOOGLE_API_KEY
    python scripts/generate_images.py --start 5 --end 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORYBOARD = ROOT / "docs" / "video-storyboard.md"
OUT_DIR = ROOT / "assets" / "images"
STYLE_SUFFIX = ", photorealistic documentary cinematography style, warm earth tones with medical green accents, 16:9 aspect ratio"


def extract_prompts(storyboard_path: Path) -> list[str]:
    """Extrait les 20 prompts des blocs ```...``` qui suivent **Prompt IA**."""
    text = storyboard_path.read_text()
    matches = re.findall(r"\*\*Prompt IA\*\* :\s*\n```\n(.*?)\n```", text, re.S)
    if not matches:
        sys.exit(f"Aucun prompt trouvé dans {storyboard_path}")
    return [m.strip() for m in matches]


def generate_imagen(client, prompt: str, out_path: Path, variants: int = 1) -> None:
    """Génère une image via l'API Imagen (nécessite google-genai + clé)."""
    response = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config={"number_of_images": variants, "aspect_ratio": "16:9"},
    )
    if variants == 1:
        response.generated_images[0].image.save(str(out_path))
    else:
        for i, img in enumerate(response.generated_images):
            stem = out_path.with_suffix("")
            img.image.save(f"{stem}-v{i+1}{out_path.suffix}")


def generate_pollinations(prompt: str, out_path: Path, seed: int = 42) -> None:
    """Génère une image via pollinations.ai (gratuit, sans clé)."""
    import urllib.parse, urllib.request, ssl

    encoded = urllib.parse.quote(prompt[:1500])  # limite URL raisonnable
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1920&height=1080&seed={seed}&model=flux&nologo=true&enhance=true"
    )
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["pollinations", "imagen"],
                        default="pollinations",
                        help="Service d'image (pollinations gratuit ou imagen avec clé API)")
    parser.add_argument("--start", type=int, default=1, help="Première scène (1-20)")
    parser.add_argument("--end", type=int, default=20, help="Dernière scène (1-20)")
    parser.add_argument("--variants", type=int, default=1,
                        help="Variantes par scène (Imagen seulement)")
    parser.add_argument("--style-suffix", default=STYLE_SUFFIX,
                        help="Suffixe ajouté à chaque prompt pour cohérence visuelle")
    parser.add_argument("--force", action="store_true",
                        help="Régénérer même si l'image existe déjà")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les prompts sans appeler l'API")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = extract_prompts(STORYBOARD)
    print(f"Storyboard parsé : {len(prompts)} prompts trouvés.")
    print(f"Provider : {args.provider}\n")

    if args.dry_run:
        for i, p in enumerate(prompts, 1):
            if args.start <= i <= args.end:
                print(f"\n--- Scène {i:02d} ---\n{p}{args.style_suffix}")
        return

    # Setup provider
    imagen_client = None
    if args.provider == "imagen":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("Erreur : GOOGLE_API_KEY non définie pour --provider imagen.\n"
                     "Obtenir une clé gratuite sur https://aistudio.google.com.")
        try:
            import google.genai as genai
        except ImportError:
            sys.exit("Module 'google-genai' manquant : pip install google-genai")
        imagen_client = genai.Client(api_key=api_key)

    n_done, n_skip, n_fail = 0, 0, 0
    for i, prompt in enumerate(prompts, 1):
        if not (args.start <= i <= args.end):
            continue
        out = OUT_DIR / f"scene-{i:02d}.png"
        if out.exists() and not args.force:
            print(f"  [skip] scene {i:02d} déjà : {out.name}")
            n_skip += 1
            continue
        full_prompt = prompt + args.style_suffix
        try:
            if args.provider == "pollinations":
                generate_pollinations(full_prompt, out, seed=42 + i)
            else:
                generate_imagen(imagen_client, full_prompt, out, variants=args.variants)
            size_kb = out.stat().st_size / 1024
            print(f"  [{i:02d}/20] ✓ {out.name} ({size_kb:.0f} KB)")
            n_done += 1
        except Exception as e:
            print(f"  [{i:02d}/20] ✗ {e}")
            n_fail += 1

    print(f"\nTerminé : {n_done} générées, {n_skip} skippées, {n_fail} échecs.")
    print(f"Dossier : {OUT_DIR}")


if __name__ == "__main__":
    main()
