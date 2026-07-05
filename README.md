# QQ Fact Check

Standalone `/事实核查` plugin split out from `astrbot_plugin_qq_agent_core`.

## Commands

- Reply to a message and send `/事实核查`.
- Send `/事实核查 要核查的内容` directly.
- English aliases in normal message text: `/factcheck`, `fact-check`.

## Behavior

- Extracts quoted text and inline text.
- Extracts up to `fact_check_max_images` image URLs from the current or quoted message.
- Uses a lightweight Gemini model to turn text/images into checkable questions.
- Optionally searches Anysearch for pre-retrieval evidence before the final grounded check.
- Uses the configured main Gemini model fallback chain for the final fact-check.
- Falls back to `这条我现在没查成。` with an optional short reason when extraction or model calls fail.

## Configuration

Managed by AstrBot WebUI through `_conf_schema.json`.

- `gemini_api_key`: Gemini API key. Empty means use `GEMINI_API_KEY`.
- `fact_check_pre_model`: pre-processing model.
- `fact_check_main_models`: ordered fallback list.
- `fact_check_max_images`: max images per request.
- `fact_check_max_image_bytes`: max bytes per image download.
- `fact_check_anysearch_enabled`: enable Anysearch pre-retrieval evidence.
- `fact_check_anysearch_api_key`: optional Anysearch API key. Empty means anonymous access or `ANYSEARCH_API_KEY`.
- `fact_check_anysearch_extract_top_urls`: number of public result pages to extract into plain-text snippets.
- `fact_check_show_failure_reason`: append a short friendly reason to failures.

## Anysearch evidence mode

When `fact_check_anysearch_enabled` is true, the plugin sends extracted checkable claims to
`fact_check_anysearch_endpoint` and injects cleaned search snippets plus a small number of public
page excerpts into the final Gemini prompt. This supplements Gemini Google Search grounding; it does
not replace the existing claim extraction, image handling, fallback, queue, cache, follow-up, or QQ
forward-message output flow.

Do not enable this mode for groups where fact-check queries may contain private data, because the
claims and extracted public URLs are sent to Anysearch.

The old bot files under `D:\Codex\QQ_Agent` and `D:\Codex\PDF_OCR` are not modified by this plugin.
