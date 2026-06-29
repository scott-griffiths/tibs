Note: This tibs logo generator code was created by an LLM based on an animated image sketched on a iPad in Cafe Nero one Saturday morning.
It is used to generate the tibs logo PNG file, and is kept here so I can easily update it if needed.

Yes, I'm spending too much time on this, but that's the beauty of non-work projects - there's no one to tell you what 
your priorities should be. A bespoke animated logo with cat art is an important part of a bit manipulation package!

# tibs transition

This package contains an editable vector rebuild of the supplied animated PNG, simplified into a one-way transition from three boxes to `tibs`.

The animation uses persistent stroke segments rather than fades: the first box becomes a three-stroke `t` plus the `i`, the `i` dot rises out of the top of its stem, the second box mostly holds position as the `b`, and the third box splits its vertical sides so the folded halves form the middle bar of the `s`.

## Best Figma path

1. In Figma, choose `Plugins > Development > Import plugin from manifest...`.
2. Select `tibs-transition-figma-plugin/manifest.json`.
3. Run `Plugins > Development > tibs transition generator`.
4. Adjust stroke, corner radius, color, spacing, size, duration, and frame count in the plugin UI, then generate.

The plugin creates editable vector frames on a new Figma page. The frames use one shared ease-in-out sine timing curve sampled into 28 frames by default. The animated PNG and browser preview hold on the first frame for `initial_delay_ms` before the transition starts. The animated PNG plays once and then holds on the final frame.

## Other files

- `tibs-transition-frames.svg`: all generated frames in a grid, importable into Figma as vectors.
- `tibs-transition-preview.html`: browser preview with live controls for stroke, radius, color, spacing, size, and duration.
- `tibs_transition_generator.py`: configurable Python generator for regenerating the package.
- `tibs-transition-preview.png`: non-editable animated PNG preview of the generated animation. It plays once and then holds on the final frame. Use this file in web pages with the normal `image/png` MIME type. The generator also copies this file to `doc/tibs.png`.


## Regenerate

Edit the hard-coded `CONFIG` block at the top of `tibs_transition_generator.py`, then run the generator with no command-line options:

```bash
python3 tibs_transition_generator.py
```
