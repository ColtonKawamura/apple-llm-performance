# Open-Weights LLM Leaderboard

Ranking of open-weights language models sourced exclusively from the [Artificial Analysis Open-Weights Leaderboard](https://artificialanalysis.ai/leaderboards/models?weights=open).

## Data

Every model and metric comes from `data/data.json`, which is a snapshot of the AA leaderboard.

### Updating

```
python3 tracker/fetch_aa.py         # fetch latest and update data.json
python3 tracker/fetch_aa.py --dry-run  # preview changes
python3 tracker/validate.py          # check data quality
python3 tracker/build.py             # rebuild docs/
```

### Schema

**Models** have these fields from AA:
- `id` — URL-safe identifier
- `name` — Display name
- `creator` — Organization that published the model
- `intelligenceIndex` — AA Intelligence Index score (higher is better)
- `costPerTask` — Cost in USD per task
- `timePerTask` — Time in minutes per task
- `latency` — Time to first token in seconds
- `contextWindow` — Maximum context window in tokens
- `aaUrl` — Link to the model on artificialanalysis.ai

**Categories** define the rankings:
1. Intelligence (by `intelligenceIndex`, descending)
2. Cost per Task (by `costPerTask`, ascending)
3. Speed / Time per Task (by `timePerTask`, ascending)
4. Latency / First Token (by `latency`, ascending)
5. Context Window (by `contextWindow`, descending)

### Model filtering

Only open-weights models are tracked. The fetcher excludes:
- Closed-weight creators: Anthropic, OpenAI, SpaceXAI
- Proprietary model patterns: Muse Wonder/Ember/Glimmer, Grok, Gemini 4, Gemini 3.x (Argon/Carbon/Hydrogen)

## Build

`docs/` is the deployable site (GitHub Pages):
- `docs/index.html` — Leaderboard page
- `docs/style.css` — Stylesheet

Run `python3 tracker/build.py` to regenerate.

## Files

| File | Purpose |
|------|---------|
| `data/data.json` | Single source of truth — models + categories |
| `tracker/fetch_aa.py` | Fetch AA leaderboard, update data.json |
| `tracker/registry.py` | Load data.json, expose ranked lists |
| `tracker/validate.py` | Validate data quality |
| `tracker/render_status.py` | Generate HTML + CSS for docs/ |
| `tracker/build.py` | Orchestrate the build |
