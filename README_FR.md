# ComfyUI KittenTTS 2 Pro — Guide français

Suite de 15 nodes pour **KittenTTS 2** (1,7 milliard de paramètres), avec TTS en français, clonage vocal, expressions et synthèse par morceaux.
Le code utilise directement le package **`kittenml`**, contrairement aux anciennes intégrations ONNX 0.8.

**Statut : prototype développeur 0.1.4**, testé avec des mocks, **pas encore avec les véritables poids du modèle**.

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


## Clonage français : éviter la transcription anglaise

- La version `kittenml==0.9.7` impose `language="en"` dans son transcripteur natif. Le node **Transcribe Reference** expose désormais `language = French` (par défaut), `Auto` ou `English` et appelle le pipeline Whisper dans le mode choisi. Cette adaptation utilise le point d'accès interne `_load()` de KittenML 0.9.7, protégé par un message d'erreur si l'API change.
- Dans **Voice Clone**, `reference_text_override` est désormais prioritaire sur le résultat Whisper connecté. Il doit contenir la transcription *exacte du seul audio conservé* par Prepare Reference, et non le texte à générer.
- Avec une référence de 10–20 s, propre et monolocuteur, commencer avec `reference_prompt_policy=default`. Tester `on` uniquement si la voix manque de fidélité, surtout pour les textes plus longs ; le prompt de référence complet peut causer des répétitions sur les textes courts.
- Pour un enregistrement déjà correct, comparer `normalize_peak=false` avec `true` ; éviter les traitements audio agressifs. `weights=packed`, `decoder=default`, `normalize_english_text=false`.
- La ressemblance ne peut pas être garantie : l'amplitude des améliorations dépend de la voix et du modèle. Tests réalisés avec mocks, pas avec la vraie RTX 3090.

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

Dépôt officiel de cette intégration communautaire : [arthoriouss/ComfyUI-KittenTTS2-Pro](https://github.com/arthoriouss/ComfyUI-KittenTTS2-Pro).
Consulte [`docs/COMFY_REGISTRY_TEMPLATE.md`](docs/COMFY_REGISTRY_TEMPLATE.md) avant la mise en ligne et remplace le `PublisherId` provisoire.
La version API ciblée est `kittenml==0.9.7` pour garder un environnement reproductible.

Le node `Transcribe Reference` possède une sortie typée `transcript_for_clone`, directement connectable au node `Voice Clone` (entrée `transcript`).

## Important — workflows d'exemple et seed

Les workflows `basic_tts.json` et `longform.json` v0.1.2 incluent `control after generate = fixed` après `seed` dans les valeurs sauvegardées. Les anciens JSON v0.1.0 pouvaient décaler les champs, avec `temperature = 0` ou `top_p = 0` et une erreur **Input out of range**. Réimporter les exemples v0.1.2 ou saisir manuellement `max_new_tokens=1000`, `temperature=1.0`, `top_p=0.95` (même avec `advanced_sampling=false`, ComfyUI valide ces champs).

## Débit de parole et variations — 0.1.3

Le modèle KittenTTS 2 **ne propose pas de paramètre natif `speed` ou `speaking_rate`**. `advanced.chunk_gap_s` réduit seulement les pauses entre morceaux et ne rend pas les mots eux-mêmes plus rapides. Pour accélérer le **débit audio**, utilisez le node `KittenTTS 2 Pro · Audio Speed / Tempo` après `Generate TTS` ou `Voice Clone`, puis avant `Preview Audio`.

- `speed = 1.20` : audio 20 % plus rapide ; `1.00` : normal ; `0.85` : plus lent.
- `preserve_pitch = true` (conseillé) : conserve la hauteur de la voix via `librosa` (un peu de traitement CPU, légère modification possible du timbre).
- `preserve_pitch = false` : accélération plus simple, mais augmente aussi la hauteur de la voix.
- Le `sample_rate` du conteneur AUDIO reste identique (24 kHz en sortie du modèle).

**La seed existe déjà** dans `KittenTTS 2 Pro · Generation Settings` et est transmise dans `advanced.seed` même quand `advanced_sampling = false`. Dans l'interface ComfyUI, réglez `control after generate = randomize` (au lieu de `fixed`) pour obtenir une nouvelle seed à chaque génération ; conservez `fixed` pour une sortie reproductible. `seed` n'est pas un contrôle de vitesse.

Workflow : `examples/faster_french_seed_variations.json`. Le nouveau node ne remplace pas l'apprentissage du modèle et ne garantit pas une qualité égale à l'audio original. Version développeur, sans validation matérielle sur RTX 3090.

## Seed directement dans Voice Clone — 0.1.4

Le node **KittenTTS 2 Pro · Voice Clone** propose désormais `seed` et `seed_mode` sans obligation de connecter `Generation Settings` :

- `seed_mode = randomize` : une **nouvelle seed à chaque Queue Prompt**, même avec les mêmes texte et référence. La sortie `generation_report` affiche `seed=... (randomize)` pour noter une variante appréciée.
- `seed_mode = fixed` : reproduit la même variante avec la valeur de `seed` (par exemple `42`). La même seed n'implique une sortie identique que si les autres paramètres et le backend sont déterministes.
- `seed_mode = settings` (valeur par défaut pour ne pas modifier les anciens workflows) : hérite de la seed transmise par `Generation Settings` quand ce node est connecté ; sinon, laisse KittenML choisir sa seed.

Si `Generation Settings` est connecté en mode `fixed` ou `randomize`, **seule sa seed est remplacée** ; le preset, la pénalité de répétition et les autres options restent actives. Les options ne sont jamais modifiées dans le node `Generation Settings` lui-même.

**Attention :** ComfyUI peut afficher un deuxième contrôle `control after generate` sous le widget numérique `seed` : laisse-le sur `fixed` quand `seed_mode=randomize`, puisque notre mode aléatoire génère déjà une seed par exécution. Tu peux aussi choisir `seed_mode=fixed` et faire varier le widget numérique avec `control after generate=randomize` ; pour une seed reproductible, garde ce contrôle sur `fixed`.

Pour les nouveaux tests de clonage, importe `examples/voice_clone.json` ou `examples/voice_clone_with_whisper.json` : les deux sont préréglés avec `seed_mode=randomize`. Si tu conserves ton ancien workflow, choisis explicitement ce mode dans le node après mise à jour et redémarrage de ComfyUI.
