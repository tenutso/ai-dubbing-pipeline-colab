# Examples

Sample assets you can use to try the pipeline and to model your own inputs.

## Files

| File | Description |
|------|-------------|
| `oqlf_glossary.txt` | A sample **OQLF (Quebec French) glossary** with common technology, business and UI terms, plus anglicisms to avoid. Pass it via `--glossary` so the translation step keeps terminology consistent. |

## Using the glossary

```bash
python dubbing_pipeline.py \
    --input inputs/your_video.mp4 \
    --glossary examples/oqlf_glossary.txt
```

The entire file is sent to Gemini as context during translation, so you can:

- Add your **brand names**, **product names** and **domain jargon**.
- Override any term you want translated a specific way.
- Keep comments (`#` lines) for your own readability — they document intent.

### Format

Each non-comment line is a simple mapping:

```
source term => OQLF French term
```

The format is intentionally human-readable; the model interprets it as guidance
rather than a strict find-and-replace, so it still handles inflection and context
naturally.

## Bring your own video

This repo does not ship a sample video (to keep it lightweight). To test:

1. Drop any short `.mp4` into an `inputs/` folder, **or**
2. Pass a Vimeo URL directly with `--input https://vimeo.com/<id>`.

Start with a 30–60 second clip to validate your credentials and setup before
processing longer content.
