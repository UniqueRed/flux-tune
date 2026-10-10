# Turn labeling

Implementation: `src/speakerturn/data/labeling.py`. Source: AMI manual word alignments (NXT 1.6.2), punctuation tokens dropped.

1. **Spurts.** A speaker's consecutive words merge when the inter-word gap is < 0.2 s.
2. **Backchannel.** A spurt shorter than 1.0 s with ≤ 3 words.
3. **Turn.** A turn starts at a non-backchannel spurt of speaker S. It covers S's later spurts until the next non-backchannel spurt of a different speaker starts (the *change*). Other speakers' backchannels do not end a turn.
4. **Turn end.** The end of S's last word before the change.
   - **Overlapped ending (spec clarification, DECISIONS D-007):** S's last spurt may straddle the change and end ≤ 1.0 s after it. That is a normal overlapped hand-over, and the turn ends at that spurt's end.
   - **Interruption:** S keeps talking more than 1.0 s past the change. These turns are excluded.
5. **Hold pauses.** Gaps between S's spurts inside a turn.

**Inclusion filter for clips:** a change exists (not the meeting end), the turn is not interrupted, it has ≥ 3 words, and it lasts 1.0–12 s. Turns that start while another speaker is still talking (`overlap_at_start`) are kept. Only the target's own headset channel enters the mixture, so the previous speaker appears only as headset bleed. This is a reported stratum.

Counts over all 171 annotated AMI meetings (2026-10-02):

| | Turns |
|---|---|
| Usable | 18,904 |
| Without overlap at start | 7,834 |
| With an overlapped ending | 6,146 |
| With ≥ 1 hold pause | 3,319 |

## Known label noise (state in every report)
- This is an operational definition. Backchannel vs. short turn is decided by thresholds.
- AMI word times are manual-ish alignments with ~10 ms resolution, plus alignment error.
- Headset channels contain low-level bleed from other participants. In a mixture, that bleed is part of the "target" signal.
- Overlapped endings are labeled at the target's speech end, not at the floor transfer.
- Only ~18% of usable turns contain a hold pause, so hold-pause false cuts are tested on a minority of turns. Hold strata are reported separately.
