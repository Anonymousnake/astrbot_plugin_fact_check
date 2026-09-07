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
cd /home/ubuntu/repos/astrbot-plugins
git status --short --branch
git pull --ff-only
git submodule update --init plugins/astrbot_plugin_fact_check
git status --short --branch
/home/ubuntu/AstrBot/.venv/bin/python -m py_compile plugins/astrbot_plugin_fact_check/*.py
sudo systemctl restart astrbot
```

Commit and push the plugin update first, then commit and push its submodule pointer in the control repository. Production loads the control repository's pinned commit through a symlink. Start from a clean workspace and restart only after the submodule update, compilation, and final clean-workspace check succeed. Preserve any existing deploy-key SSH URL overrides.

## Config and data policy

- Real config: `data/config/astrbot_plugin_fact_check_config.json`
- Runtime data: `data/plugin_data/astrbot_plugin_fact_check/`
- Do not commit secrets, cookies, local paths, cache files, database files, or generated media.
