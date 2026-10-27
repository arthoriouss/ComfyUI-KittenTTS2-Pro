# ComfyUI KittenTTS 2 Pro — Guide français

Suite de 14 nodes pour **KittenTTS 2** (1,7 milliard de paramètres), avec TTS en français, clonage vocal, expressions et synthèse par morceaux.
Le code utilise directement le package **`kittenml`**, contrairement aux anciennes intégrations ONNX 0.8.

**Statut : prototype développeur 0.1.2**, testé avec des mocks, **pas encore avec les véritables poids du modèle**.

## Installation Windows Portable

1. Fermer ComfyUI.
2. Copier le dossier `ComfyUI-KittenTTS2-Pro` dans `ComfyUI_windows_portable/ComfyUI/custom_nodes/`.
3. Lancer `install_portable_windows.bat` (indispensable : le Manager ne pose pas automatiquement le backend KittenML afin de préserver CUDA).
4. Redémarrer ComfyUI, chercher `KittenTTS 2 Pro`.
5. Glisser `examples/basic_tts.json` sur le canvas.

L'installateur conserve la version Torch/CUDA et torchaudio de ComfyUI. En cas de conflit, privilégier une **installation ComfyUI séparée**, plutôt que de forcer les dépendances.

## Réglages pour RTX 3090 (24 Go)

- `device=cuda`, `backend=pytorch`, `weights=packed`, `decoder=default` ; première génération lente lors du téléchargement et du chargement.
- `voice_name=French`, `normalize_english_text=false` pour des phrases en français.
- Test TTS d'abord, clonage ensuite avec 5–30 secondes d'une seule voix, sans musique.
- Préciser la **transcription exacte** du clip si possible ; sinon la V2 peut utiliser Whisper automatiquement.
- Pour l'expressivité, choisir `preset=expressive`, puis éventuellement les réglages avancés ; les émotions sont encore en beta.

## Les principales chaînes

`Load Model → Generate TTS → Preview Audio`

`Load Audio → Prepare Reference → Voice Clone → Preview Audio` (+ Load Model)

`Load Model → Longform Chunks → Preview Audio` ; les morceaux arrivent à la fin de l'exécution du node, pas en diffusion temps réel.

## Limites et cadre légal

Le projet n'enregistre pas les références en permanence ; le clip WAV temporaire est effacé après l'appel de génération.
Ne cloner que les voix pour lesquelles vous avez une autorisation. Les **poids du modèle** ont une licence distincte du code : [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md).

Pour les problèmes d'installation : [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Test GPU réel

ComfyUI fermé, à partir du dossier du projet :

```powershell
..\..\..\python_embeded\python.exe scripts\smoke_test.py --device cuda --weights packed --voice French
```

La commande crée `kitten-smoke.wav` et affiche le temps de génération et le RTF. Elle télécharge les poids officiels si nécessaires.

## Publication

Ce projet est destiné à un nouveau dépôt GitHub `ComfyUI-KittenTTS2-Pro`, mais rien n'a été publié à ta place.
Consulte [`docs/COMFY_REGISTRY_TEMPLATE.md`](docs/COMFY_REGISTRY_TEMPLATE.md) avant la mise en ligne et remplace le `PublisherId` provisoire.
La version API ciblée est `kittenml==0.9.7` pour garder un environnement reproductible.

Le node `Transcribe Reference` possède une sortie typée `transcript_for_clone`, directement connectable au node `Voice Clone` (entrée `transcript`).

## Important — workflows d'exemple et seed

Les workflows `basic_tts.json` et `longform.json` v0.1.2 incluent `control after generate = fixed` après `seed` dans les valeurs sauvegardées. Les anciens JSON v0.1.0 pouvaient décaler les champs, avec `temperature = 0` ou `top_p = 0` et une erreur **Input out of range**. Réimporter les exemples v0.1.2 ou saisir manuellement `max_new_tokens=1000`, `temperature=1.0`, `top_p=0.95` (même avec `advanced_sampling=false`, ComfyUI valide ces champs).

## Clonage en français — nouveauté 0.1.2

- **Transcribe Reference** propose désormais `French` (par défaut), `Auto` et `English`. Dans KittenML 0.9.7, Whisper force l'anglais ; le node appelle désormais directement son pipeline chargé pour imposer la langue choisie. Cette API interne pourrait changer dans une future version.
- `reference_text_override` a maintenant priorité sur la transcription Whisper connectée. Écris-y **les paroles exactes de l'audio de référence conservé**, et non le texte à générer.
- **Voice Clone** : `reference_prompt_policy=default` recommandé ; compare éventuellement avec `on` si la fidélité est médiocre, surtout sur des phrases longues. Un contexte audio forcé peut provoquer des répétitions sur de courts textes.
- Référence propre de 10 à 20 secondes, une seule personne ; `packed`, décodeur `default`, normalisation anglaise désactivée. La fidélité n'est pas garantie et les tests avec vrais poids RTX 3090 restent à faire.
