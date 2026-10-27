# Responsible use

Use only voice recordings you own or are permitted by the identifiable speaker to clone.
Do not impersonate, mislead, evade consent or distribute synthetic audio as authentic speech.
Inform listeners when audio is synthetic where appropriate. Do not embed reference WAVs in workflows distributed publicly.
Reference audio is held in process memory and written only to an OS temporary directory for an inference/transcription call;
this package deletes that temporary file after the call. The upstream KittenML package may cache downloaded model files.
