Note: This tibs logo generator code was created by an LLM based on an animated image sketched on a iPad in Cafe Nero one Saturday morning.
It is used to generate the tibs logo PNG file, and is kept here so I can easily update it if needed.

Yes, I'm spending too much time on this, but that's the beauty of non-work projects - there's no one to tell you what 
your priorities should be. A bespoke animated logo with cat art is an important part of a bit manipulation package!

# tibs transition

This package contains an editable vector rebuild of the supplied animated PNG, simplified into a one-way transition from three boxes to `tibs`.

The animation opens on three hollow rounded boxes and holds there for `initial_delay_ms` (1 second by default). It then uses persistent stroke segments rather than fades: the first box becomes a three-stroke `t` plus the `i`, the `i` dot rises out of the top of its stem, the second box mostly holds position as the `b`, and the third box splits its vertical sides so the folded halves form the middle bar of the `s`.

## Outputs

Run `logo_generator.py` to generate the animated PNG, a static PNG of the final frame, and a browser preview.

The morph starts at full speed and eases out into the final frame, sampled into 28 frames by default. The animated PNG plays once and then holds on the final frame.

The preview and rendered PNG support a configurable border inside the logo. Configure border size and border color in the preview controls. Increasing the border size does not change the logo's outer dimensions. Set border size to `0` to disable the border.

## Files

- `tibs-transition-preview.html`: browser preview with live controls for stroke, radius, color, spacing, size, and duration.
- `tibs-transition-preview.png`: non-editable animated PNG preview of the generated animation. It plays once and then holds on the final frame. Use this file in web pages with the normal `image/png` MIME type. The generator also copies this file to `doc/tibs.png`.
- `tibs-final.png`: static PNG of the final `tibs` frame. The generator also copies this file to `doc/tibs_static.png`.


## Regenerate

Edit the hard-coded `CONFIG` block at the top of `logo_generator.py`, then run the generator with no command-line options:

```bash
python3 logo_generator.py
```
