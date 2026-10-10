# Findings and design notes

The engineering log behind flux-tune, condensed from the project's decision log. IDs (D-0xx) match
references in code comments. Numbers marked "demo" come from an earlier 40-call AMI set that was
superseded by the speaker-disjoint benchmark in D-026; that set is not included in this repo.

Before flux-tune, the project tried a speaker-conditioned end-of-turn model (a voiceprint fed into
an open turn detector, so it would ignore background talkers). A pre-registered pilot showed the
speaker signal in short overlapped windows was too weak to help (AUC about 0.62 for the oracle
voiceprint), so that idea was closed as a negative result and the project pivoted to measuring
and tuning Flux itself.

- **D-007 (2026-10-02) Turn-end rule for overlapped hand-overs.** In AMI, about half of raw turns overlap the next speaker's start, so "last word before the change" is ill-defined there. An overlapped ending (target stops ≤ 1.0 s after the change) ends at the target's speech end. Longer straddles are interruptions and are excluded. Result: 18,904 usable turns, versus 5,324 under the strictest reading. See `docs/labeling.md`.
- **D-008 (2026-10-02) AMI audio via HTTP range requests.** Headset WAVs have a 44-byte PCM header, so only the needed byte ranges are fetched (turn segments ± context, and enrollment segments). Full channels are never downloaded or stored.
- **D-020 (2026-10-06) Day-1 validation results** (34 AMI val clips, 54 turns, 3 passes, $0.17).
  - **Determinism:** before the first EOT, manual-mode and real-run confidence streams are identical (same values, same 240 ms update times).
  - **Offline re-thresholding is approximate, not exact.** Flux fires EndOfTurn at 80 ms resolution (e.g. awe 1.84 between updates at 1.68 and 1.92), but reports confidence only every 240 ms. Simulated first-EOT minus real first-EOT at θ=0.7: mostly 0 to +0.18 s, with 22/24 single-turn clips within 0.25 s and 2/24 large misses (a crossing on a hidden frame). Sessions show more count mismatches (state-machine divergence after the first EOT). The bias direction undercounts cut-offs, so **recommendations must be confirmed with real passes**.
  - **Hindsight labels:**
    - Raw boundary agreement with AMI manual turn ends is mediocre: median −0.10 s, MAE 0.52 s, 52% within 200 ms. Word-confidence filtering does not help.
    - For tuning they are good. Across 7 settings, cut-off rate rank correlation vs. human labels is 0.991, and median-latency rank correlation is 0.929. Per-turn cut-off classification agrees 83–93%.
    - They undercount cut-offs by 0–9 points, which is reported as a known bias.
  - **Observation (not a claim):** on AMI meeting speech, Flux at the default θ=0.7 fires before the human-labeled turn end in 39% of turns (human labels; 30% hindsight), mostly in mid-turn pauses. At θ=0.85 it is 24%. Meeting speech is not Flux's target domain, and n is small.
- **D-021 (2026-10-06) Gap-based hindsight labels undercount cut-offs about 2×; agent-times mode added.** Earlier demo set (40 calls, 160 human-labeled turns; not included in this repo).
  - Gap hindsight finds 176 turns: holds of 2 s or more split a turn in two.
  - Rank correlation with human labels across real thresholds is 1.0 for cut-off rate and 1.0 for median latency. But the absolute cut-off rate is too low: 3.4% vs 6.9% at θ=0.94, 6.8% vs 13.1% at 0.89, 23.9% vs 34.4% at 0.7. So gap-mode recommendations miss the target on human labels.
  - Fix: `--agent_times` (agent speech intervals, from a stereo agent channel or agent timestamps) delimits turns by speaker change, matching the human definition. Gap mode stays for mono audio, with a calibration warning in the report.
  - In the demo, agent intervals are the constructed gaps between turns. Caveat: with recordings of an existing bot, agent onsets are that bot's own decisions.
- **D-022 (2026-10-06) Noise stress result** (earlier demo set, report half, 90 turns, background talker = AMI train-split raw headset speech).
  - At the default θ=0.7, TIR 10 dB: TER(1.5 s) rises from 0.13 to 0.63 (Δ +0.50 [+0.39, +0.61]), and 31% of turns get no EOT before the next user turn. TIR 0 dB: TER 0.66, Δ +0.53 [+0.40, +0.65].
  - At θ=0.94: cut-offs go to 0, but no-decision is 0.89–0.92.
  - Threshold tuning cannot fix background speech; it only trades cut-offs for extensions.
  - The inserted-word metric was normalized per user-speech minute (~200+/min), which overstates it. It is changed to per audio minute.
- **D-023 (2026-10-06) The exact curve comes from real passes on a fixed grid; simulation is preview only.** In agent-times mode on multi-turn audio, offline re-thresholding drifted badly from real runs (at θ=0.94, simulated median latency 3.06 s vs real 1.77 s). With no resets between turns in manual mode, the simulated state machine diverges. flux-tune now runs real passes at REAL_GRID = {0.6, 0.7, 0.8, 0.86, 0.9, 0.94, 0.98} (≈ $0.0077/audio-min each) and recommends per target cut-off rate (5/10/20%), chosen on the TUNE half and reported on the REPORT half.
  - Demo (AMI meeting speech, 40 calls, agent-times labels, report half 79 turns): default 0.7 gives 36.7% cut-offs at 0.54 s median latency. Recommended for 10%: θ=0.94, giving 10.1% [3.8, 17.7] at 1.77 s. For 20%: θ=0.86, giving 15.2% at 1.20 s. The 5% target (θ=0.98) gives 6.3% [1.3, 11.4] on the report half.
  - Noise at the default, TIR 10 dB: ΔTER +0.60 [+0.47, +0.73]; 85 inserted words per audio-minute.
  - Caveat for every claim: AMI is spontaneous *meeting* speech with long holds, not agent-directed speech.
- **D-024 (2026-10-06) Voice lab: a local browser app for recording agent-style speech** (`scripts/voice_lab.py`, `src/speakerturn/voicelab/`).
  - Runs only on localhost: published artifacts cannot use the microphone, and the Deepgram key must stay server-side.
  - Agent prompts are spoken with Flux TTS (`POST /v2/speak?model=flux-haley-en`, verified 200; Aura-2 fallback). They are cached on disk, and TTS spend goes into the shared ledger at $0.045 per 1k characters.
  - Mic audio (16 kHz int16, 80 ms chunks) goes to the local server, which relays it live to Flux (θ=0.7, display only) and saves it.
  - Agent-start/end markers travel in-band with the audio, so turn boundaries are stamped at exact audio times. In a smoke test they landed within one 80 ms chunk of the true times.
  - **The agent advances only when the user presses Next, never on Flux's EndOfTurn**, so the recordings are not shaped by the threshold under test.
  - The mic is zeroed while the agent speaks (default on, toggleable), and headphones are recommended.
  - Analysis (`scripts/analyze_lab.py`):
    - Quiet sessions → flux-tune with a turn-level tune/report split (few long files) and the synthetic background-talker test.
    - Background sessions → scored as real-world evidence at the recommended and default thresholds.
  - Recordings are gitignored and never leave the machine.
- **D-025** Voice lab results on the author's own voice (2026-10-07)
  - Data: 1 quiet session (10 answers, 179 s) and 1 real-background session (10 answers, 138 s; podcast from a separate speaker, Mac built-in mic, mic zeroed while the agent speaks).
  - Quiet, threshold picked on the tune half (5 turns), reported on the other 5: 0.94 → 0/5 cut-offs vs 2/5 at default 0.7; median latency 1.28 s vs 0.34 s. Paired difference in cut-off rate −0.4 [−0.8, 0.0]: the CI touches zero, so with 5 report turns this is suggestive, not significant.
  - Real background, 10 turns: at 0.7, 20% cut-offs, 25% waits >1.5 s, 10% never ended. At 0.94: 0% cut-offs, but 70% waits >1.5 s and 40% never ended. Background speech keeps the confidence below a strict threshold, so the quiet-room fix makes the noisy room worse. This matches D-022 (AMI): threshold tuning cannot fix background speech; a speaker-aware front end is needed.
  - Podcast words merged into the user's transcripts (e.g. "So I've just landed in San Francisco" inside a spelled surname).
  - Fix along the way: `run_batch` now retries on websockets ConnectionClosed (one network drop killed the first run; cached passes were reused).
- **D-026** Larger benchmarks: AMI with speaker-disjoint tune/report, plus SpokenWOZ phone calls (2026-10-09)
  - Why: the voice-lab result (D-025) is 10 turns from one speaker. More data was needed, without recruiting other people.
  - AMI benchmark (`scripts/make_ami_bench.py`):
    - Tune on 416 turns from 28 AMI dev speakers. Report on 787 turns from 43 AMI test speakers in different meetings.
    - No turn is reused, and each speaker is capped (15 tune / 20 report turns).
    - The AMI test split had never been used by flux-tune.
  - Bootstrap: CIs now resample whole speakers (`--groups`; `bootstrap.py::_sampler`), not turns. The tune/report split is fixed by `--split_map`.
  - AMI report half:

    | Setting | Cut-offs [CI] | Strict cut-offs | Median latency [CI] |
    |---|---|---|---|
    | Default 0.7 | 26.2% [22.0, 30.6] | 22.9% | 0.56 s |
    | 0.94 (chosen for ≤10%) | 5.2% [3.0, 7.7] | 3.6% | 1.70 s [1.62, 1.84] |

    Paired difference (0.94 vs default): cut-offs −21.0 pts [−24.5, −17.6], median latency +1.14 s [+1.12, +1.27]. The target is met on unseen speakers.
  - AMI noise test (report half, background talker at 10 dB TIR):
    - At 0.7: 61% of turns kept open >1.5 s; 20% never end.
    - At 0.94: 90% never end.
    - About 80–100 background words per audio-minute enter the transcript.
    - Confirms D-022 at 5× the size.
  - SpokenWOZ (`scripts/fetch_spokenwoz.py`, CC BY-NC 4.0):
    - 39 test dialogues, customer channel at 8 kHz, agent turns taken from annotations.
    - SNG0653 was excluded because its audio is truncated (29 s of audio vs 91 s of annotations).
    - Split by call: 19 tune / 20 report. CIs resample calls.
    - Report half:

    | Setting | Cut-offs [CI] | Strict cut-offs | Median latency |
    |---|---|---|---|
    | Default 0.7 | 15.1% [11.2, 20.1] | 6.3% | 0.34 s |
    | 0.86 (chosen for ≤10%) | 8.2% [6.0, 10.8] | 2.8% | 0.65 s |
    | 0.94 | 6.0% | — | 1.20 s |

  - Cross-corpus finding: the best threshold depends on the audio. Meeting speech needs 0.94 for ≤10%, while phone calls need 0.86. Using the meeting setting on phone calls would add about 0.55 s of median latency for about 2 pts fewer cut-offs.
  - Strict count (new `fcr_words_after`):
    - It counts only cut-offs where the speaker said at least one more word.
    - On SpokenWOZ, over half of the cut-offs fire during the final word. Phone-band word timings are likely looser, so the reports show both counts.
  - Considered and skipped:
    - Smart Turn clips: isolated clips with no conversation, and it is unclear how the incomplete ones were made.
    - ICSI: meetings again, overlapping with AMI.
  - Report builder fixes:
    - The noise section is now conditional.
    - `--caveat` and `--group_noun` were added.
    - The label check is no longer attached to every report by default (the voice-lab report had wrongly shown the AMI demo's label check).
  - Spend: $23.14 of $40. The cached reruns cost $0.
