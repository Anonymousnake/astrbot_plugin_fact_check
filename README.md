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
- Uses the configured main Gemini model fallback chain for the final fact-check.
- Falls back to `这条我现在没查成。` with an optional short reason when extraction or model calls fail.

## Configuration

Managed by AstrBot WebUI through `_conf_schema.json`.

- `gemini_api_key`: Gemini API key. Empty means use `GEMINI_API_KEY`.
- `fact_check_pre_model`: pre-processing model.
- `fact_check_main_models`: ordered fallback list.
- `fact_check_max_images`: max images per request.
- `fact_check_max_image_bytes`: max bytes per image download.
- `fact_check_show_failure_reason`: append a short friendly reason to failures.

The old bot files under `D:\Codex\QQ_Agent` and `D:\Codex\PDF_OCR` are not modified by this plugin.
