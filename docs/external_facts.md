# External Facts

## Deepgram Flux

> **Verified on 2026-10-02** against primary docs at developers.deepgram.com (raw `.md` versions of each page) and deepgram.com/pricing. Every fact cites its source. Short quotes are verbatim.

Key sources (abbreviations used below):
- **[REF]** API reference (AsyncAPI spec): https://developers.deepgram.com/reference/speech-to-text/listen-flux
- **[QS]** Quickstart: https://developers.deepgram.com/docs/flux/quickstart
- **[CFG]** End-of-turn parameters: https://developers.deepgram.com/docs/flux/configuration
- **[STATE]** State machine: https://developers.deepgram.com/docs/flux/state
- **[CONF]** Configure message: https://developers.deepgram.com/docs/flux/configure
- **[FET]** ForceEndTurn: https://developers.deepgram.com/docs/flux/force-end-turn
- **[BYO]** Bring your own turn detection: https://developers.deepgram.com/docs/flux/own-turn-detection
- **[CLOSE]** CloseStream: https://developers.deepgram.com/docs/flux/close-stream
- **[MIG]** Nova-3 migration: https://developers.deepgram.com/docs/flux/nova-3-migration
- **[CMP]** Flux vs Nova-3: https://developers.deepgram.com/docs/flux/flux-nova-3-comparison
- **[RL]** Rate limits: https://developers.deepgram.com/reference/api-rate-limits
- **[PRICE]** Pricing: https://deepgram.com/pricing
- **[AUTH]** Authentication: https://developers.deepgram.com/guides/fundamentals/authenticating

### 1. Endpoint, model, auth
- **Endpoint:** `wss://api.deepgram.com/v2/listen` (server `wss://api.deepgram.com/`, channel `/v2/listen`) [REF]. Flux must use `/v2/listen`, not `/v1/listen` [QS]. There are also regional hosts: `api.eu.`, `api.au.` and `api.in.deepgram.com` [RL]. The India Flux URL is `wss://api.in.deepgram.com/v2/listen` (https://developers.deepgram.com/changelog/2026/9/15.md).
- **Models:** `flux-general-en` (English) and `flux-general-multi` (10 languages: en, es, fr, de, hi, ru, pt, ja, it, nl) [REF], https://developers.deepgram.com/docs/models-languages-overview. `model=flux` is listed as a mistake [QS].
- **Auth:** the `Authorization` header is `Token YOUR_DEEPGRAM_API_KEY` [AUTH]. The WebSocket example sends `headers = {"Authorization": "Token YOUR_DEEPGRAM_API_KEY"}` [BYO]. Short-lived tokens are available via the Auth token API [AUTH].

### 2. Audio format and chunking
Source: [QS], [MIG].

| Type | Encoding | `encoding`/`sample_rate` params | Sample rates |
|---|---|---|---|
| Raw | `linear16`, `linear32`, `mulaw`, `alaw`, `opus`, `ogg-opus` | **Required** ("`16000` recommended") | `8000`, `16000`, `24000`, `44100`, `48000` |
| Containerized | linear16 in WAV; opus in Ogg; opus in WebM | **Omit** | Auto-detected |

- **Channels:** "Mono only" [MIG].
- **Chunk size:** "**80ms audio chunks strongly recommended** for optimal model performance and latency" [QS]. At 16 kHz linear16, 80 ms is 2560 bytes (code comments in [QS] and https://developers.deepgram.com/docs/flux/agent).
- **Keepalive:** there is no KeepAlive message. Nova-3's KeepAlive is "Replaced by pings, 60s timeout" [CMP].

### 3. Message schema
**Server to client** [REF]: `Connected`, `TurnInfo`, `ConfigureSuccess`, `ConfigureFailure`, `Warning` (the connection stays open), `Error` (fatal).
**Client to server** [REF]: binary audio (Media), `CloseStream`, `ForceEndTurn`, `Configure`.

- `Connected`: `type`, `request_id` (uuid), `sequence_id` (int, starts at 0 and increments across *all* server messages).
- `TurnInfo` required fields [REF]: `type`, `request_id`, `sequence_id`, `event`, `turn_index` (int), `audio_window_start`, `audio_window_end` (seconds), `transcript` ("Text that was said over the course of the current turn", so it is cumulative per turn), `words[]`, `end_of_turn_confidence` ("Confidence that no more speech is coming in this turn").
  - Optional fields: `trigger`, which is on every `EndOfTurn` and only there. Values are `model`, `manual` (ForceEndTurn) or `timeout` (`eot_timeout_ms`), and it is an open enum. `languages` and `languages_hinted` appear only with `flux-general-multi`.
  - `event` enum: `Update`, `StartOfTurn`, `EagerEndOfTurn`, `TurnResumed`, `EndOfTurn`.
  - `words[]` items: `word` and `confidence` are required. `start` and `end` (double, seconds) are in the schema but **not required** [REF]. Word-level timestamps shipped on the hosted API on 2026-07-02 (https://developers.deepgram.com/changelog/2026/7/2.md) and in self-hosted release 260630 (https://developers.deepgram.com/changelog/2026/6/30.md). Before then, words had only `word` and `confidence`. The [FET] example response still shows words without `start`/`end`.
  - The schema types `confidence`, `audio_window_*` and `end_of_turn_confidence` as `string` with title "float", but the examples show JSON numbers [REF], [STATE]. Parse defensively.
- `Error`: `type:"Error"`, `sequence_id`, `code` (e.g. `INTERNAL_SERVER_ERROR`, `UNPARSABLE_CLIENT_MESSAGE`), `description` [REF], [CONF].
- `Warning`: `code` (SCREAMING_SNAKE_CASE, e.g. `FORCE_END_TURN_NO_ACTIVE_TURN`) and `description` [REF], [FET].

**Verbatim example TurnInfo** [MIG]:
```json
{
  "type": "TurnInfo",
  "request_id": "2ba892a1-6c0d-4d92-9b89-0000000000",
  "event": "Update",
  "turn_index": 0,
  "audio_window_start": 0,
  "audio_window_end": 0.47999996,
  "transcript": "",
  "words": [...],
  "end_of_turn_confidence": 0.0009,
  "sequence_id": 2
}
```
**Verbatim StartOfTurn excerpt with word timestamps** [STATE]:
```json
{
  "event": "StartOfTurn",
  "turn_index": 0,
  "audio_window_start": 0.0,
  "audio_window_end": 0.6,
  "transcript": "Hi I",
  "words": [
    { "word": "Hi", "confidence": 0.95, "start": 0.42, "end": 0.56 },
    { "word": "I", "confidence": 0.92, "start": 0.56, "end": 0.6 }
  ],
  "end_of_turn_confidence": 0.1
}
```
**State-machine guarantees** [STATE]:
- `EagerEndOfTurn` always has a non-empty transcript.
- `TurnResumed` only follows `EagerEndOfTurn`.
- The `EndOfTurn` transcript always equals the immediately preceding `EagerEndOfTurn` transcript. If the transcript changes, `TurnResumed` is emitted first.
- `turn_index` increments immediately after `EndOfTurn`.
- Every `StartOfTurn` has a non-empty transcript, and it is the recommended barge-in trigger.
- By default only `Update`, `StartOfTurn` and `EndOfTurn` are emitted. `EagerEndOfTurn`/`TurnResumed` appear only when `eager_eot_threshold` is set [MIG], [CFG].

### 4. Parameters (query string)
Sources: [CFG], [QS], [REF].

| Param | Range | Default | Notes |
|---|---|---|---|
| `eot_threshold` | 0.5–1.0 | 0.7 | `1.0` suppresses native EndOfTurn (the timeout still applies) |
| `eager_eot_threshold` | 0.3–0.9 | unset (eager off) | Must be ≤ `eot_threshold`, otherwise an error |
| `eot_timeout_ms` | 500–60000 | 5000 | "Maximum silence duration before forcing an EndOfTurn, regardless of confidence". The timer resets on new speech |
| `keyterm` | repeatable string | none | Plain terms, no weights (`term:0.15` is silently treated as literal). Up to 100 terms via Configure [CONF]. Keyterm doc: 500-token limit (https://developers.deepgram.com/docs/keyterm) |
| `language_hint` | repeatable | none | `flux-general-multi` only |
| `profanity_filter` | true/false | false | [REF] |
| `numerals` | true/false | false | Can be toggled mid-stream since 2026-09-25 |
| `redact` | `numbers`, `aggressive_numbers` | none | Only these on Flux. PII/PCI/PHI return 400 (changelog 2026-06-24) |
| `mip_opt_out`, `tag` | any | — | [REF] |
| `encoding`, `sample_rate`, `model` | see §1–2 | — | |

Typed as "Float (passed as string in URL or SDK)" [CFG]. Do not pass `language=` [QS].

### 5. Is end_of_turn_confidence continuous?
**Yes.** `end_of_turn_confidence` is a *required* field on every `TurnInfo`, including every `Update` [REF]. Cadence: "`Update` messages are sent for approximately every 0.25 seconds of transcribed audio, regardless of transcript updates, unless a state change has occurred" [STATE]. This is ~250 ms of *audio*, not wall-clock time. The docs do not give an exact 240 ms figure. The [MIG] example shows `audio_window_end: 0.47999996` (a multiple of 0.08 s, which fits 80 ms internal frames, but that is inference only). With ForceEndTurn, the reported confidence is "the model's actual confidence … not inflated to 1.0" [FET].

### 6. Control messages (exact JSON)
- **ForceEndTurn** [FET]: `{"type": "ForceEndTurn"}`. It ends the turn on the audio transcribed so far, with no extra decode pass, and emits `EndOfTurn` with `"trigger": "manual"`. If no turn is active, it is ignored and returns `Warning` `FORCE_END_TURN_NO_ACTIVE_TURN`. If an Eager EOT is pending, it supersedes it with no TurnResumed. Messages are processed in strict order. Added 2026-08-28 (https://developers.deepgram.com/changelog/2026/8/28.md).
- **Bring your own turn detection** [BYO]: set `eot_threshold=1.0`, leave `eager_eot_threshold` unset, and send `ForceEndTurn` when your detector fires. `eot_timeout_ms` (max 60000) remains a backstop (`trigger:"timeout"`). Example: `wss://api.deepgram.com/v2/listen?model=flux-general-en&eot_threshold=1.0&eot_timeout_ms=30000`.
- **Configure** [CONF]: thresholds must be nested under `thresholds`, and omitted fields are unchanged.
  ```json
  {"type": "Configure",
   "thresholds": {"eager_eot_threshold": 0.4, "eot_threshold": 0.7, "eot_timeout_ms": 6000},
   "keyterms": ["apple", "banana", "orange"]}
  ```
  Other fields: `language_hints` (an empty array `[]` clears them) and `numerals` (bool). The `keyterms` list *replaces* the existing list (`[]` clears it). Changes apply immediately in stream order, and already-transcribed audio is not reprocessed. Changes are atomic: all fields apply or none do.
  - On success the server replies with `ConfigureSuccess` (the full active config).
  - On a validation failure it sends `ConfigureFailure` (`code`/`description`) and the stream keeps going.
  - On a *schema* failure it sends `type:"Error"` with code `UNPARSABLE_CLIENT_MESSAGE` and **closes the connection**.
- **CloseStream** [CLOSE]: `{"type": "CloseStream"}`. Flux decodes the audio already received, emits `Update`s, then closes. It **does not emit EndOfTurn**, there is no summary metadata, and the close comes "without sending a WebSocket close status code". To finalize a turn first, send `ForceEndTurn` and then `CloseStream` [FET].

### 7. Target speaker / diarization / speaker conditioning
**None is supported (this confirms the hypothesis).** "Speaker Diarization | Flux 🚫" [CMP]. The API reference query params have no speaker, diarize, enrollment or channel parameter [REF]. Audio must be mono [MIG]. The Deepgram CLI changelog says "`--diarize` warns and is cleared consistently, since Flux does not support it" (https://developers.deepgram.com/changelog/2026/8/19.md). The only conditioning inputs are `keyterm`/`keyterms` and `language_hint` [REF]. Use cases marked 🚫 for Flux include meeting transcription, pre-recorded audio and call analytics [CMP].

### 8. Pricing and limits
- **Pricing** [PRICE], streaming, per audio-minute. The page is labeled "Limited-time promotional rates on streaming" and gives no end date.

  | Model | Pay As You Go | Growth |
  |---|---|---|
  | Flux English | **$0.0065/min** (regular $0.0077/min) | $0.0057/min (regular $0.0065/min) |
  | Flux Multilingual | $0.0078/min | $0.0068/min |

  Flux English PAYG works out to about $0.39/hr at the promo rate and $0.462/hr at the regular rate. The hourly figures are computed here, not quoted.
- **Concurrency** [RL]:
  - Pay As You Go: Flux STT streaming "Up to 150 concurrent requests" in all regions (NA, EU, AU, IN).
  - Growth: 225 in NA and 150 elsewhere.
  - Limits apply **per project**. Extra projects on a self-serve account are limited to 1 concurrent stream, and spreading traffic across projects violates the ToS.
- The launch-promo (October 2025) cap was 50 concurrent connections (https://developers.deepgram.com/changelog/2025/10/2.md). That is historical.

### 9. Sending audio faster than real time
- No Flux-specific statement exists. The general streaming guide says: "Because Deepgram allows audio to be streamed at a **maximum of 1.25x realtime**, if you send a large buffer of audio, the stream may wind up being significantly delayed" (https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio).
- The official Flux agent example nonetheless sends a WAV file in 4096-byte (~128 ms) chunks with `asyncio.sleep(0.01)` between them, which is about 12x real time (https://developers.deepgram.com/docs/flux/agent). It does not say whether the server throttles this or whether it changes turn-detection timing.
- **Treat any faster-than-1.25x behavior on Flux as unverified.** Note that `eot_timeout_ms` is defined as silence duration, presumably in audio time, which is not stated.

### 10. Relevant changelog entries
Source: https://developers.deepgram.com/changelog/{yyyy/m/d}.md
- **2025-10-02:** Flux launched (`flux-general-en`, `/v2/listen`). It was free during October 2025, with up to 50 concurrent connections.
- **2025-10-15:** self-hosted Flux. "Flux must be run in its own self-hosted deployment".
- **2025-10-16:** expanded audio formats beyond linear16 (raw encodings plus WAV/Ogg containers).
- **2026-01-13:** WebM/Opus container support.
- **2026-02-27:** `Configure` control message (thresholds and keyterms mid-stream).
- **2026-04-16:** Flux Multilingual on self-hosted (10 languages, code-switching).
- **2026-06-24:** number redaction on Flux.
- **2026-07-02:** **word-level `start`/`end` timestamps** added to `words[]`.
- **2026-07-17:** `numerals` added (connection-time only). **2026-09-25:** `numerals` can be toggled via Configure.
- **2026-08-28:** **`ForceEndTurn`, the `trigger` field, `eot_threshold=1.0` fully-manual mode**, and per-turn mode switching via Configure.
- **2026-09-14:** SDKs add `send_force_end_turn()` (Python v7.9.0) and `sendForceEndTurn()` (JS v5.11.0).

### Contradictions / caveats found in the docs
1. **Encodings:** the [QS] prose says "Flux only accepts `linear16`", but its own table and the [REF] spec list 6 raw encodings plus containers. That line is stale from before the 2025-10-16 change.
2. **Profanity filter:** [CMP] marks Profanity Filtering 🚫 for Flux, while [REF] lists `profanity_filter` and the feature overview lists it as available (https://developers.deepgram.com/docs/flux/feature-overview). The 2026-05-21 changelog also adds it for Flux multilingual.
3. **Word timestamps:** they are optional in the schema and absent in the [FET] example. They are present since 2026-07-02, but clients should tolerate missing `start`/`end`.
4. **Numeric types:** float fields are typed `string` in [REF] but are numbers in every example.
5. **Update cadence:** the docs say "approximately every 0.25 seconds" of audio, not an exact 240 ms.
6. **Latency claim:** "~260ms end-of-turn detection (p50 at defaults)" [MIG] is a vendor claim.
