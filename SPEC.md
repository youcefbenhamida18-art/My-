# MangaForge v2 — architecture & data contract (internal, English)

Goal: one **2-minute anime-style short per day** (7 shorts are joined into a weekly ~14 min episode). Target look: cinematic
anime (heavy camera work, parallax, particles, glow, grade, real voice acting, scored audio) built from AI key-art that is
rendered through **Blender**, with an optional GPU tier (image-to-video) for hero shots. Everything must **fail loudly** or
degrade *visibly and logged* — v1's silent fallbacks (grey gradient art, broken Arabic text) are the bugs we are fixing.

## 0. Hard rules for every module
* Python 3.10+ (CI runs 3.11, sandbox 3.13). `from __future__ import annotations`. Type hints welcome, no heavy frameworks.
* Core deps available everywhere: `numpy, scipy, Pillow, opencv (cv2), requests, PyYAML`. Heavy/optional libs
  (`torch, transformers, rembg, onnxruntime, edge_tts, google api`) are imported **lazily inside functions** and every
  caller has a working fallback + a log line saying which path was taken.
* Never `pip install` (no network in the dev sandbox). Never hit a real network service in tests — mock HTTP
  (e.g. `http.server` on localhost in a thread, or `unittest.mock`). Real API behaviour is verified from docs, not guessed.
* Config only through `from mangaforge.config import cfg, log, ...` (see `mangaforge/config.py`). Shared vocabulary only from
  `mangaforge/schema.py`. Camera math only from `mangaforge/camera_paths.py`.
* Paths inside JSON are **absolute strings**. Work dir for a job: `config.work_dir(job)`; do not write outside it
  (except tests → pytest `tmp_path`). Temp files in tests/experiments go to the scratchpad dir you were given.
* Errors: raise a typed exception with an actionable message (what failed, what the user should set/install). Warnings go to
  `log.warning`. Never print secrets (keys/tokens).
* Each module has a `if __name__ == "__main__"` self-demo OR a pytest file `tests/test_<module>.py` that runs offline in < 60 s.
* Do not edit files you do not own. Do not touch `_v1_reference/` (read-only copy of the v1 code for porting ideas).

## 1. Pipeline & artifacts (per job = one short; `W = work_dir(job)`)

```
story.py/manga.py ─► W/script.json ─┬► voice.py ─► W/voice/*.wav (+ durations)
                                    └► art/      ─► W/art/<image_id>.png, depth, backdrop, sway, rim   (+ optional i2v clips)
director.plan(script, voices, art) ─► W/shots.json          (render-ready list of "shot" dicts, §3)
render (Blender | cv fallback)     ─► W/frames/shot_XXXX/f_####.png  (or raw video)
post.finish_shot(shot, raw)        ─► W/shots/shot_XXXX.mp4   (graded, bloomed, grained, transitions)
sound.build_soundtrack + subs.write_ass + assemble ─► out/short_<job>.mp4   (+ optional vertical cut)
assemble.weekly(7 shorts)          ─► out/episode_NNN.mp4
```
Stages are resumable: a stage skips work whose output already exists (hash/sentinel), so a crashed CI shard re-runs cheaply.

## 2. `script.json` (output of story.py / manga.py, input of art/voice/director) — see `schema.normalize_script`
```jsonc
{ "version": 2,
  "meta": {"title_ja","title_ar","description_ar","summary","tags":[..],"hashtags":[..],"part":3,"of":7,"episode":12},
  "characters": [{"id":"kaito","name_ja","name_ar","gender":"male|female","voice":"hero|villain|elder|young|narrator|girl|boy","look":"english booru-style tags"}],
  "style": "english style tags for every image prompt",
  "images": [ {"id":"i01","scene":"s1","prompt":"english: setting, action, camera angle, lighting (no text)",
               "characters":["kaito"],"framing":"wide|medium|closeup","palette":"cold blue night","no_humans":false,
               "source_path": null | "/abs/panel.png"} ],      // source_path set => use this art, do not generate (manga mode)
  "shots":  [ {"id":1,"image":"i01","type":"establishing|wide|medium|closeup|insert|action|impact|reaction|transition",
               "dur":3.0, "framing":{"zoom":1.0,"cx":0.5,"cy":0.5},
               "camera":{"move":"push_in|…","strength":0.5}, "fx":["petals","wind_sway"], "grade":"calm",
               "line":{"speaker":"kaito|narrator","ja":"…","ar":"…","emotion":"angry","intensity":3} | null,
               "sfx":["whoosh"], "music":"tense", "trans_in":"cut", "notes":""} ] }
```
Several shots may reuse one image with different `framing` (cut-ins) — this is how real anime reuses art.

## 3. Render-ready shot dict (output of `director.plan`, stored in `shots.json`)
```jsonc
{ "idx": 7, "id": 8, "scene": "s2", "type": "closeup",
  "dur": 3.25, "frames": 78, "fps": 24, "step": 2,          // frames = round(dur*fps); renderer makes ceil(frames/step) drawings, each held `step` frames
  "size": [1920,1080],
  "plate": { "image":"/abs.png", "backdrop":"/abs.jpg", "depth":"/abs.png",       // depth: 8-bit gray, 255 = NEAREST
             "sway":"/abs.png"|null, "rim":"/abs.png"|null,                       // 8-bit masks, same aspect as image
             "subject": {"x":0.5,"y":0.55,"depth":0.8},                           // focus point (image coords, y down) + its depth value
             "sequence": null | {"dir":"/abs","pattern":"f_%04d.png","frames":97,"fps":24} },   // AI-animated plate (replaces the still as texture)
  "framing": {"zoom":1.0,"cx":0.5,"cy":0.5},                                      // initial crop window on the plate (image coords, y down)
  "camera": {"move":"push_in","strength":0.6,"dof":true,"shake":0.0,"hit_at":null,"focal":35},   // hit_at seconds or null
  "fx":      [{"kind":"petals","amount":0.6,"at":0.0,"dur":null}],                // FX_BLENDER kinds only; at/dur seconds (dur null = until end)
  "post_fx": [{"kind":"flash","at":0.0,"dur":0.2}, {"kind":"impact_frame","at":0.35,"dur":0.08}],   // FX_POST kinds only
  "grade":   {"preset":"battle","bloom":null,"grain":null,"vignette":null,"chroma":null,"contrast":null,"saturation":null},  // nulls = preset default
  "trans_in":"flash", "trans_out":"cut",
  "seed": 481516,
  "text":  {"ar":"…","ja":"…","speaker":"kaito","emotion":"angry"} | null,
  "audio": {"voice":"/abs.wav"|null, "voice_start":0.2, "voice_dur":2.4, "sfx":[{"kind":"boom","t":0.0}], "music":"battle", "intensity":4},
  "paths": {"frames_dir":"/abs/W/frames/shot_0007", "raw":"/abs/W/raw/shot_0007.mkv", "final":"/abs/W/shots/shot_0007.mp4"} }
```
Camera/plate geometry shared by engines: the frame covers `camera_paths.PLATE_MARGIN` (0.90) of the plate at `dz=0` and
`framing.zoom=1`; `framing.zoom>1` crops in (camera starts closer), `cx,cy` pick the crop centre. Plate and frame are 16:9.

## 4. Module APIs (owners in brackets)

### [me] `llm.py`
`ask_json(prompt, *, images=None, temperature=1.0, max_tokens=30000) -> dict|list` — Gemini transport with model discovery, key
rotation, JSON repair; `images` = list of `bytes` (JPEG/PNG) sent as inline_data (vision). Tests mock it.

### [A] Blender — `mangaforge/render/`
* `blender_scene.py` — runs **inside Blender** (`blender -b --factory-startup -noaudio -P blender_scene.py -- --job job.json`).
  Standalone file: only `bpy, mathutils, numpy` + stdlib; it may `sys.path.insert` the project root to
  `import mangaforge.camera_paths`. `job.json = {"shot": <§3 shot dict>, "settings": {"engine":"CYCLES|EEVEE","samples":24,
  "threads":2,"device":"CPU|GPU","denoise":true,"motion_blur":true,"scale":1.0,"resume":true}}`.
  Writes contiguous `frames_dir/f_0001.png … f_NNNN.png` (one per drawing, NOT per output frame) and `frames_dir/done.json`
  `{"drawings":N,"engine":..,"blender":"5.2.1","warnings":[..],"features":{"plate":true,"wind":true,"petals":true,…}}`.
* `blender_runner.py` (host): `find_blender() -> str|None` (cfg.blender_bin, PATH, common install dirs, snap/flatpak),
  `blender_version(bin) -> str|None`, `render_shots(shots, *, engine="auto", on_progress=None) -> dict[int, Path]`
  (idx → frames_dir; resumable; per-shot retry; clear errors; thread/sample settings from `cfg`),
  `selftest(work) -> dict` (renders a tiny synthetic shot with *every* feature, returns per-feature OK/FAIL + writes
  `selftest_report.json` and `contact_sheet.png`).
* `render/__init__.py` re-exports `find_blender, render_shots, selftest, BlenderNotFound`.

### [B] Art — `mangaforge/art/` and `mangaforge/manga.py`
* `art/__init__.py`: `generate_art(script, work, *, progress=None) -> dict[str, dict]` image_id → `{"image","backdrop","depth","sway","rim","subject":{x,y,depth},"provider","seed"}`; `class ArtGenerationError(RuntimeError)`; `provider_status() -> dict`.
  **No silent placeholder**: if the provider fails for an image after retries → try the next provider; if everything fails raise `ArtGenerationError`
  unless `cfg.allow_placeholder_art` (then a clearly-labelled procedural placeholder is produced and logged at ERROR).
* `art/providers.py`: `Provider` base + `HordeProvider` (AI Horde, async API, model auto-discovery, anime post-processing), `PollinationsProvider`,
  `ComfyProvider` (ComfyUI HTTP API, built-in SDXL txt2img graph), `A1111Provider` (`/sdapi/v1/txt2img`), `CloudflareProvider` (v1 behaviour), `get_provider(name) / auto chain`.
* `art/prompts.py`: anime prompt builder (quality tags per model family, character look injection, negative prompt, per-character stable seeds).
* `art/layers.py`: `build_layers(image_path, out_dir, stem) -> dict` (depth via Depth-Anything-V2 if `transformers+torch` present else a robust CV/edge-aware fallback; sway & rim masks; blurred backdrop; subject point; upscale/sharpen to `cfg.art_target_w`).
* `art/i2v.py`: `animate(image_path, prompt, out_dir, *, seconds, fps, seed) -> {"dir","pattern","frames","fps"}|None` via ComfyUI API with a user workflow template (placeholders), returns None when `cfg.i2v_backend=="off"`.
* `manga.py`: `ingest(path, work, *, max_pages=None) -> script` for folders/CBZ/PDF of manga/manhwa/webtoon pages: panel detection (Gemini vision via `llm.ask_json` + classical CV fallback), reading order, dialogue extraction (ja/ko/zh/en → Japanese+Arabic lines), output normalised with `schema.normalize_script` (images have `source_path` = cropped panels).

### [C] Audio, subtitles, post — `mangaforge/{voice,sound,subs,post}.py`
* `voice.synthesize(script, work) -> dict[int, {"path","dur","speaker","provider"}]` keyed by shot id (only shots with a `line`). Providers: VOICEVOX HTTP engine (preferred; emotion → style, per-character speaker), edge-tts, `human` override (`assets/voices/<shot id>.*`), `silent` (placeholder tone-free silence, logged ERROR). `available_providers() -> dict`.
* `sound.build_soundtrack(shots, out_wav, *, seed=0) -> Path`: timeline = cumulative `dur`; voice from `shot["audio"]`; music per shot `audio.music` (user tracks in `assets/music/<mood>/` else procedural score: koto/shamisen-style plucks, taiko, pads, drones), SFX from `audio.sfx` (user files in `assets/sfx/<kind>*.wav|mp3` else procedural), ducking under voice, light limiter, stereo 44.1 kHz WAV, loudness ≈ −16 LUFS.
* `subs.py`: **Arabic subtitles done right**. `write_ass(shots, out_path, *, size, vertical=False, show_ja=False) -> Path` (logical text → libass shapes/bidi itself; do NOT pre-shape), `render_line_png(text, size, ...) -> PIL.Image` (Pillow: use Raqm with `direction='rtl', language='ar'` on raw text when `features.check('raqm')`, otherwise arabic_reshaper+bidi with `Layout.BASIC`), `find_arabic_font() -> (path, family)` (looks in `assets/fonts`, then system), `self_check() -> dict`.
* `post.py`: `GRADE_PRESETS` (one per `schema.GRADES`), `grade_params(preset, overrides) -> dict`, `build_filtergraph(shot, in_fps) -> str` (pure/testable), `finish_shot(shot, raw, in_fps=None) -> Path` (raw = PNG-sequence dir or video; applies bloom, grade, chroma, grain, vignette, flash/impact_frame/glitch, transition fades; duplicates drawings to `fps`; exact `frames` length), `vertical_crop_filter(shot) -> str`.

### [me] story/director/pipeline/assemble/cv_engine/doctor/CLI
`story.make_bible()`, `story.write_week_outline()`, `story.write_short(bible, state, part)` → `script`; `director.plan(script, voices, art, job) -> list[shot]`;
`render/cv_engine.py` (no-Blender fallback with the same shot dict); `assemble.assemble_short(...)`, `assemble.weekly(...)`; `pipeline.py`; `__main__.py` CLI; `doctor.py`.

## 5. Quality bar (what "done" means for each stream)
* Offline tests pass; code paths that need the network/Blender/GPU are exercised through mocks AND clearly marked in the report as "not run for real".
* Every optional dependency absent ⇒ module still imports and the fallback path is exercised in a test.
* Report back (≤ 400 words): files written, public API actually implemented (diff vs this spec), what was verified how, what could NOT be verified, risks.

---
# ADDENDUM (v2.1) — Blender feature tiers, the user's "100 Blender capabilities" list, extras contract

The user supplied a list of 100 Blender capabilities and wants every produced video to pass through as many of them as is
*useful*, plus more. Honest rule: **a feature counts only if it really runs in the render and is reported**. Each shot's
`done.json` carries `"features": {key: "ok" | "off" | "skipped:<why>" | "failed:<error>"}` and `python -m mangaforge features`
aggregates it against the 100-item registry (`mangaforge/render/features.py`, owned by me). Never mark a feature "ok" unless
the code path completed. A failing feature must never abort the shot (wrap each in `with ctx.feature("key"):`).

## 6. Tiers
* **Tier 1 – always on, must be rock-solid** [A1 `blender_scene.py`]: depth-mesh plate (true displacement) + blurred backdrop, emission-only
  unlit shading with alpha layers, procedural shader wind/heat-haze/water-ripple (UV warp driven by Noise/Voronoi nodes + sway mask), rim glow,
  analytic particles (petals/leaves/embers/snow/dust/fireflies/rain/sparks as instanced quads, keyed per drawing), moving fog layers, fake god-rays,
  speed lines, slash arc, shockwave ring, lightning, all 17 camera moves, dolly zoom (focal keys), **rack focus** (animated focus distance
  between two depths), DoF, handheld + impact shake, motion blur, OIDN denoise, persistent data, colour-management view transform,
  resumable frames, optional `.blend` save with relative paths.
* **Tier 2 – on when the shot asks for it (or director adds it), each independently guarded** [A2 `blender_extras.py`]: 3D foreground dressing
  (pebbles/grass/leaves) scattered with **Geometry Nodes** (fallback: python instancing), procedural **terrain** ground (noise-displaced grid),
  Subdivision/Solidify/Bevel/Simple-Deform modifiers on props, toon shading (Toon BSDF in Cycles, ShaderToRGB ramp in EEVEE) + inverted-hull outlines
  (Backfacing trick, works in both engines), procedural bump, the **plate lights the 3D props through GI** (Light Path "Is Camera Ray" trick, set up by A1,
  consumed by A2), contact shadows, glass shards + lens drops (Glass BSDF, real refraction of the plate), anamorphic/lens-flare streak planes, debris
  shards (analytic ballistic + ground bounce; own Voronoi-like fragments, not the removed Cell Fracture add-on), aura (rim-mask energy glow + rising wisps),
  swirl (elemental ribbon spirals for "breathing technique" style moves), glTF/FBX/OBJ/USD asset import hook (`shot["asset3d"]`), HDRI world if
  `assets/hdri/*` exists, 3D extruded title text (Latin/Japanese; Arabic is a Pillow-rendered image plane because Blender cannot shape Arabic),
  optional Freestyle lines for props.
* **Tier 3 – heavy sims, only when `settings["physics"]` is true** [A2]: Mantaflow smoke + fire (`smoke`, `fire`), liquid splash (`splash`), cloth flag/cape
  (`cloth_flag`), real rigid-body debris, real particle systems + wind/turbulence/vortex **force fields**, real scattering volume + spot light
  (`volume_fog`, also `settings["volumetrics"]`). Small domains, replay cache, hard frame caps; clean up and report `failed:` instead of hanging.
* **Post (ffmpeg, robust)** [C]: bloom, grade, contrast/saturation curves, chromatic/lens-distortion look, film grain, vignette, **anamorphic streak flares**
  from highlights, flash, impact frames, glitch, transitions, **time remapping** (`hit_stop`, `speed_ramp` via a drawing→frame map), stock-element
  `overlay` with chroma/luma key or screen blend (`assets/overlays/*`), letterbox, vertical crop.
* **Assembly** [me]: ffmpeg concat + audio + ASS subtitles (default); optional Blender-VSE assembler (guarded, 4.x `sequences` / 5.x `strips` shim);
  optional Blender-compositor post (guarded; 5.x `compositing_node_group` shim).

## 7. Runtime rules for `blender_scene.py`
1. Build everything, then loop **every** frame `f = 1..frames` calling `scene.frame_set(f)` in order and render only when `f` is a drawing frame
   (`(f-1) % step == 0`) and the PNG does not already exist (resume). Stepping every frame is what makes particles/cloth/fluids/rigid bodies deterministic.
2. Output names are contiguous by drawing index: `f_0001.png` = frame 1, `f_0002.png` = frame `1+step`, …
3. Engine ids by introspection (`bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items`): `BLENDER_EEVEE_NEXT` (4.2–4.5) vs `BLENDER_EEVEE` (≤4.1, 5.x); never hard-code.
4. `Material.use_nodes` / `World.use_nodes` are deprecated in 5.x → set inside try/except; never rely on them being writable.
5. Do not touch `Action.fcurves` (removed in 5.0): compute easing in python and key densely with `keyframe_insert`; sockets/props animated via `keyframe_insert`.
6. Blender-embedded python is 3.11–3.13; `numpy` is bundled; nothing else. CLI flags only `-b --factory-startup -noaudio -P … -- …` (+ `--threads`).
7. Cycles settings for the emission-only look: `transparent_max_bounces ≥ 16`, `max_bounces 4–6`, samples from settings, `use_persistent_data`,
   OIDN denoiser if available, `film_filter 1.2–1.5`; Standard view transform unless `settings["view_transform"]` says otherwise.
8. Every extra must clean up after itself on failure (remove partial objects) and report `failed:<error>`.

## 8. Extras contract (A1 builds `ctx`, A2 consumes it) — `blender_extras.apply(ctx)`
`ctx` is a plain object with these attributes (A1 guarantees them; A2 must not assume anything else):
```
bpy, mathutils                       # modules
scene, camera, collection, world     # camera object at its start pose; new objects must be linked to `collection`
shot, settings                       # §3 shot dict ; runtime settings dict ({"engine","samples","physics","volumetrics","geonodes","view_transform",...})
engine                               # "CYCLES" | "EEVEE"
fps, step, frames, drawings          # drawings = list of blender frame numbers that will be rendered
rng                                  # random.Random(seed)
D_REF = 10.0                         # world distance from the (un-dollied) camera to the subject plane; camera looks down -Z, +Y up, starts at cam0
cam0                                 # mathutils.Vector start position (framing zoom/centre already applied)
frame_w, frame_h                     # world size of the visible frame at D_REF (dz=0, zoom=1); plate_w = frame_w/0.9
subject                              # (x, y, depth_value) of the shot subject in world units at the D_REF plane
palette                              # {"avg","top","bottom","left","right","accent"} each (r,g,b) linear 0..1 — from the plate image
ground_y                             # world y of the ground line at the D_REF plane (from plate.ground_y, default 0.85 of image height)
wind                                 # (wx, wy) shot wind vector, |w| in 0..1
cam_pose(frame)                      # camera_paths.total_pose(...) dict for that blender frame
feature(key)                         # context manager: records ok / failed:<error> in done.json["features"][key]; swallows exceptions
warn(msg)                            # collected into done.json["warnings"]
link(obj)                            # link object to the collection
```
`apply(ctx)` reads `ctx.shot["fx"]` (list of `{"kind","amount","at","dur",…}`) and `ctx.shot.get("asset3d")`, and for each Tier-2/3 kind present
(or auto-enabled kinds listed in `ctx.shot.get("auto_dress")`, e.g. `["pebbles","grass"]` chosen by the director for outdoor ground scenes) builds it
inside `with ctx.feature(key)`. Placement rule: foreground dressing lives in the band **between camera and plate**, z ∈ [cam0.z−7, cam0.z−1.5] (so DoF blurs it).

## 9. Feature keys (use exactly these strings in `features`)
A1: `plate_depth_mesh plate_backdrop emission_unlit alpha_layers wind_warp heat_haze water_ripple rim_glow particles_analytic fog_layers light_rays speed_lines
slash_arc shockwave lightning camera_moves dolly_zoom rack_focus dof handheld_shake impact_shake motion_blur denoise_oidn persistent_data view_transform
gi_plate_light resume_cache save_blend sequential_stepping addons_enabled`
A2: `scatter_geonodes scatter_python ground_terrain props_modifiers toon_shader outline_shell bump_procedural contact_shadow glass_shards lens_drops lens_flare_streak
debris_analytic aura swirl asset_import hdri_world title_text3d freestyle_lines usd_export debris_rigidbody particles_blender force_field_wind smoke_fire_mantaflow
liquid_splash cloth_flag volume_real`
C (post, reported by `post.finish_shot` in `<final>.json`): `post_bloom post_grade post_curves post_chroma post_grain post_vignette post_anamorphic post_flash
post_impact_frame post_glitch post_transitions post_time_remap post_hit_stop post_overlay_key post_letterbox`
Me: `assemble_ffmpeg vse_assembly compositor_post weekly_episode headless_batch ci_sharding`

## 10. Contract deltas
* `plate` also carries `"palette": {...}` and `"ground_y": 0.0..1.0` (image y, down). B computes both in `layers.build_layers`; A1 falls back to computing palette from the image and `ground_y=0.85`.
* shot gets `"wind": [wx, wy]`, `"physics": bool`, optional `"auto_dress": [...]`, optional `"asset3d": {"file","scale","pos":[x,y,z],"rot":[..],"toon":true}`.
* `fx` entries may carry kind-specific keys: `x,y` (image coords for flares/aura/lightning origin), `color` ([r,g,b] 0..1), `dir` (degrees), `speed`, `clip`/`mode`/`opacity` (overlay).
* `post_fx` entries: `hit_stop {at,dur}`, `speed_ramp {at,dur,speed}`, `anamorphic {amount,color}`, `overlay {clip,mode:"screen|key|alpha",at,dur,opacity}`, `flash`, `impact_frame`, `chromatic`, `glitch`.
  `post.time_map(shot) -> list[int]` (drawing index for every output frame) implements hit-stop/ramps in python so ffmpeg only sees a plain numbered sequence.
* `shot["camera"]` may also carry `"rack": null | {"from": <depth value 0..1>, "to": <depth value 0..1>}` (rack focus between two depths over the shot; absent = focus stays on `plate.subject.depth`). `plate.parallax` (optional float, default 0.35) scales depth→distance. `plate.backdrop_scale` (default 1.5): the backdrop image covers 1.5× the plate (B reflect-pads + blurs so the camera never sees an edge); A1 sizes the backdrop plane accordingly.
* `mangaforge/llm.py` is DONE (ask_json + parse_json). Tests patch `mangaforge.llm.ask_json`.
