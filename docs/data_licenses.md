# Data licenses

| Source | License | How it's obtained | What this repo contains |
|---|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (University of Edinburgh) | CC BY 4.0 | Headset audio slices fetched from the official Edinburgh server with HTTP range requests (`scripts/fetch_ami_segments.py`); full recordings are never downloaded | `data/manifests/ami_turns.jsonl` (turn boundaries built from the AMI manual word alignments, see `docs/labeling.md`) and the cached Flux outputs for the benchmark audio in `data/flux_cache` (transcripts and turn events, no audio) |
| [SpokenWOZ](https://spokenwoz.github.io/SpokenWOZ-github.io/) test set, HF `ssz1111/SpokenWOZ-Test-Audio-Fixed` and `-Text-Fixed` | CC BY-NC 4.0 (non-commercial, attribution) | Ungated on Hugging Face; only the needed dialogues are extracted from the 2.18 GB zip by range requests (`scripts/fetch_spokenwoz.py`) | Results only (`reports/spokenwoz`), which quote short transcript snippets. No audio |
| Voice lab recordings | The author's own | Recorded locally | Nothing. Recordings are gitignored |

Attribution: AMI Meeting Corpus, Carletta et al. (2005), CC BY 4.0. SpokenWOZ, Si et al. (NeurIPS 2023 Datasets and Benchmarks), CC BY-NC 4.0.

One SpokenWOZ dialogue (SNG0653) is excluded because its audio is truncated (29 s of audio against 91 s of annotations).

Checked and not used:
- ICSI Meeting Corpus: CC BY 4.0, ungated, but multi-party meetings again, overlapping with AMI.
- Pipecat smart-turn-data-v3.2: CC BY 4.0. Isolated complete/incomplete clips with no conversation context.
- Seamless Interaction: CC BY-NC 4.0, very large.
- CANDOR: gated. Fisher and Switchboard: LDC (paid).
