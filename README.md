# QQ Fact Check

Standalone `/事实核查` plugin split out from `astrbot_plugin_qq_agent_core`.

## Commands

- Reply to a message and send `/事实核查`.
- Send `/事实核查 要核查的内容` directly.
- English aliases in normal message text: `/factcheck`, `fact-check`.
- Send `/事实核查状态` to view persisted aggregate success, partial-result, failure, cache, and latency counters.

## Behavior

- Extracts quoted text and inline text.
- Extracts up to `fact_check_max_images` image URLs from the current or quoted message.
- Uses a lightweight Gemini model to turn text/images into checkable questions.
- Uses Gemini 2.5 Flash with Google Search grounding to collect evidence and produce a complete fallback result.
- Uses Gemini 3 Flash without native grounding for multi-claim and high-risk topics by default; ordinary single claims use the grounded result directly.
- Optionally searches Anysearch for extra pre-retrieval evidence before the grounded check.
- Formats replies as plain QQ-friendly text with explicit per-point `结论：` lines.
- Maps Gemini grounding support back to individual claim blocks and marks claims without direct support.
- Validates that the rendered claim still matches the requested claim before accepting a verdict.
- Requires stronger evidence for legal, medical, financial, safety, and other high-risk claims: one primary source or two independent sources.
- Detects explicit source conflicts and prevents them from becoming high-confidence conclusions.
- Keeps Anysearch evidence attached to its originating claim and rejects extracted pages that do not materially overlap that claim.
- Uses numbered source references consistently between claim hints and the final clickable source list.
- Automatically shortens cache lifetime for breaking-news and recent-event claims.
- Coalesces identical in-flight requests so concurrent users share one pipeline run.
- Tracks QQ delivery success separately from model/pipeline success.
- Preserves malformed JSON state as a `.corrupt-*` file before starting with safe defaults.
- Preserves structurally complete claim blocks when a model response is truncated instead of discarding the whole result.
- Saves cache hits as full fact-check sessions, so replying to cached results still supports follow-up.
- Falls back to segmented OneBot text when merged-forward sending fails.
- Accepts images only from trusted local adapter paths or public HTTP(S) URLs; `file://`, `base64://`,
  localhost, and private-network URLs are ignored as user-supplied URLs.
- Falls back to `这条我现在没查成。` only when no usable claim block can be recovered.

## Configuration

Managed by AstrBot WebUI through `_conf_schema.json`.

- `gemini_api_key`: Gemini API key. Empty means use `GEMINI_API_KEY`.
- `fact_check_pre_model`: pre-processing model.
- `fact_check_evidence_model`: grounded evidence-retrieval model, normally `gemini-2.5-flash`.
- `fact_check_verdict_models`: evidence-only verdict editors, normally `gemini-3-flash-preview`.
- `fact_check_verdict_policy`: defaults to `risk_based`; use `always` only when every request needs a second verdict pass.
- `fact_check_verdict_timeout_seconds`: short timeout for the Gemini 3 review; the grounded 2.5 result is sent immediately when it expires or returns no readable text.
- `fact_check_max_images`: max images per request.
- `fact_check_max_image_bytes`: max bytes per image download.
- `fact_check_anysearch_enabled`: enable Anysearch pre-retrieval evidence.
- `fact_check_anysearch_api_key`: optional Anysearch API key. Empty means anonymous access or `ANYSEARCH_API_KEY`.
- `fact_check_anysearch_extract_top_urls`: number of public result pages to extract into plain-text snippets.
- `fact_check_show_failure_reason`: append a short friendly reason to failures.
- `fact_check_session_store_enabled`: persist owner-scoped follow-up sessions across restarts.
- `fact_check_access_control_fail_open`: keep disabled so an ACL import failure does not expose the command globally.

## Quality and runtime notes

- `fact_check.py` remains the synchronous evidence pipeline. AstrBot-facing runtime coordination and configuration translation live in `runtime.py` and `pipeline_config.py`.
- The hard timeout bounds how long the bot waits. Python cannot forcibly stop an already-running worker thread, so upstream HTTP calls still retain their own bounded timeouts.
- The quality corpus under `tests/fixtures/` covers source conflicts, high-risk source strength, unrelated evidence, and ordinary low-risk claims. It intentionally does not implement user-feedback learning.

## Anysearch evidence mode

When `fact_check_anysearch_enabled` is true, the plugin sends extracted checkable claims to
`fact_check_anysearch_endpoint` and injects cleaned search snippets plus a small number of public
page excerpts into the final Gemini prompt. This supplements Gemini Google Search grounding; it does
not replace the existing claim extraction, image handling, fallback, queue, cache, follow-up, or QQ
forward-message output flow.

Do not enable this mode for groups where fact-check queries may contain private data, because the
claims and extracted public URLs are sent to Anysearch.

The old bot files under `D:\Codex\QQ_Agent` and `D:\Codex\PDF_OCR` are not modified by this plugin.
