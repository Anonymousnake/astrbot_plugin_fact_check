# astrbot_plugin_fact_check local development

This plugin is prepared to live as an independent Git repository.

## Local workflow

1. Edit and test the plugin under `D:\Codex\AstrBot\data\plugins\astrbot_plugin_fact_check`.
2. Keep real runtime config in `D:\Codex\AstrBot\data\config`.
3. Commit only plugin source, schema, docs, fixed assets, and `config.example.json`.

## Server workflow

The server copy should live at:

```text
/home/ubuntu/AstrBot/data/plugins/astrbot_plugin_fact_check
```

Update it with:

```bash
cd /home/ubuntu/AstrBot/data/plugins/astrbot_plugin_fact_check
git pull
if [ -f requirements.txt ]; then /home/ubuntu/AstrBot/.venv/bin/pip install -r requirements.txt; fi
sudo systemctl restart astrbot
```

## Config and data policy

- Real config: `data/config/astrbot_plugin_fact_check_config.json`
- Runtime data: `data/plugin_data/astrbot_plugin_fact_check/`
- Do not commit secrets, cookies, local paths, cache files, database files, or generated media.
