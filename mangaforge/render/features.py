"""`python -m mangaforge features [--job JOB] [--json]` - where each of the 100 Blender capabilities from the brief stands in THIS pipeline.

An honest map, not a brochure. Status:

    yes   implemented and used by the Blender engine (reported per shot in done.json under the `keys` listed here)
    alt   the same visual result is achieved another way (the note says how)
    post  done by the ffmpeg post stage, which runs after Blender / the cv engine on every shot
    opt   implemented but off by default: needs a switch, a GPU, ComfyUI, or an add-on
    no    NOT implemented
    na    a manual authoring tool of Blender's interface (sculpting, painting ...): meaningless in an unattended pipeline

`--job JOB` adds what really happened in that job's shots (ok / off / skipped:<why> / failed:<error> per key, from done.json).
IMPORTANT: the Blender-side code is tested here against a faithful fake `bpy` (tests/fake_bpy.py) that validates node sockets, enums and
the call order, not against a real Blender: run `python -m mangaforge selftest` on a machine with Blender once, it takes a minute.
"""
from __future__ import annotations

import collections
import json

from ..config import work_dir

STATUS = {"yes": "منفَّذ", "alt": "بديل مكافئ", "post": "في ffmpeg", "opt": "اختياري", "no": "غير منفَّذ", "na": "أداة يدوية"}

CATEGORIES = [
    "1. النمذجة والبيئات ثلاثية الأبعاد", "2. المواد والتظليل", "3. الأنيميشن والحركة", "4. الفيزياء والمحاكاة", "5. الرندر والإضاءة",
    "6. المونتاج والـ Compositing", "7. بايثون والأتمتة", "8. المؤثرات البصرية (VFX)", "9. خط الإنتاج والتوافق", "10. اللمسات السينمائية",
]

# (category index, status, English name, Arabic name, feature keys in done.json, note)
ITEMS = [
    # ---- 1 modelling / environments
    (0, "na", "Sculpting", "النحت العضوي", [], "نحت يدوي. البديل: شبكة عمق تُبنى من خريطة العمق المقدَّرة للّوحة"),
    (0, "alt", "Geometry Nodes", "عقد الهندسة", ["ground_dressing", "grass"], "العشب والحصى والأوراق شبكات تُبنى بـnumpy/bmesh بدل عقد الهندسة"),
    (0, "na", "Dyntopo", "الطوبولوجيا الديناميكية", [], "أداة نحت يدوية"),
    (0, "no", "Hard-surface modelling", "النمذجة الصلبة", [], "لا مجسمات مصنوعة يدوياً (الشخصيات والأسلحة من الرسم)"),
    (0, "yes", "Subdivision surface", "التقسيم الشبكي", ["plate_depth_mesh"], "شبكة اللوحة مقسّمة بكثافة تتبع العمق"),
    (0, "yes", "Modifiers", "المعدِّلات", ["plate_depth_mesh", "cloth_simulation"], "تُستخدم لشبكة اللوحة والقماش وأشرطة الأقواس"),
    (0, "na", "Symmetry", "التناظر", [], "أداة نحت/نمذجة يدوية"),
    (0, "na", "Retopology", "إعادة البناء الطوبولوجي", [], "أداة يدوية"),
    (0, "alt", "Grease Pencil modelling", "الرسم الخطي في الفراغ", ["curve_strokes"], "Cycles لا يرسم Grease Pencil، فالضربات (قطع/برق/دوامة) منحنيات مشطوفة متوهجة. "
                                                                                    "تتبّع حبر المانغا موجود في lineart.py لكنه غير موصول بالمشهد بعد"),
    (0, "no", "Landscape generation", "توليد التضاريس", [], "الخلفية لوحة مرسومة مع عمق؛ لا تضاريس مولَّدة"),
    # ---- 2 shading / texturing
    (1, "yes", "Shader nodes", "عقد التظليل", ["plate_unlit_emission", "toon_shading", "wind_sway", "heat_haze"], "مادة اللوحة وطبقات التأثيرات كلها عقد تظليل"),
    (1, "na", "Texture painting", "الرسم على المجسم", [], "أداة يدوية؛ الألوان تأتي من لوحة الذكاء الاصطناعي"),
    (1, "yes", "Procedural textures", "الخامات الإجرائية", ["wind_sway", "water_ripple", "sprite_atlas_procedural"], "ضجيج/فورونوي للرياح والماء وأطلس السبرايتات"),
    (1, "yes", "Toon / NPR shading", "تظليل الأنمي", ["toon_shading", "freestyle_ink"], "N·L عبر ColorRamp ثابت (يعمل في Cycles وEEVEE) + خطوط Freestyle"),
    (1, "no", "Normal & bump maps", "الخرائط الطبيعية", [], "غير مستخدمة"),
    (1, "alt", "UV unwrapping", "فرد UV", ["plate_depth_mesh"], "إسقاط مستوٍ لشبكة اللوحة والسبرايتات"),
    (1, "yes", "True displacement", "الإزاحة الحقيقية", ["plate_depth_mesh", "depth_jump_face_drop"], "إزاحة الرؤوس بحسب العمق (لا في المادة) مع حذف الوجوه عند القفزات العميقة"),
    (1, "yes", "Emission shaders", "المواد المتوهجة", ["plate_unlit_emission", "rim_glow", "fx_billboards"], "اللوحة نفسها Emission كي تبقى ألوانها بلا تغيّر"),
    (1, "alt", "Glass & refraction", "الزجاج والانكسار", ["glass_shards"], "شظايا زجاج سبرايت؛ بلا انكسار فيزيائي"),
    (1, "yes", "Alpha blending", "الطبقات الشفافة", ["fx_billboards"], "شبكتان (إضافية + ألفا) تحملان آلاف الرقاقات"),
    # ---- 3 animation / rigging
    (2, "no", "Armatures & rigging", "الهياكل العظمية", [], "لا شخصيات ثلاثية الأبعاد؛ الحركة من الكاميرا والتأثيرات والعيون"),
    (2, "no", "Inverse kinematics", "IK", [], "غير مستخدم"),
    (2, "alt", "Graph editor curves", "محرر المنحنيات", ["camera_moves"], "منحنيات الكاميرا (تسارع/تباطؤ) تُحسب في camera_paths.py وتُفتَّح كإطارات مفتاحية"),
    (2, "alt", "Shape keys", "الـ Shape Keys", ["eye_patches"], "تعابير العيون (رمش، نظرة، بؤبؤ، دموع) عبر رقعات صور لا shape keys"),
    (2, "no", "NLA editor", "التحريك غير الخطي", [], "غير مستخدم"),
    (2, "no", "Action constraints", "قيود الحركة", [], "غير مستخدمة"),
    (2, "no", "Motion tracking", "تتبع الحركة", [], "غير مستخدم"),
    (2, "no", "Weight painting", "رسم الأوزان", [], "لا هياكل عظمية"),
    (2, "alt", "Face rigging", "نظام الوجه", ["eye_patches"], "منصة العيون فقط (eye_anim.py)؛ لا تحريك فم"),
    (2, "no", "Camera tracking / matchmoving", "تتبع الكاميرا", [], "غير مستخدم"),
    # ---- 4 physics
    (3, "opt", "Cloth simulation", "محاكاة القماش", ["cloth_simulation", "cloth_flag"], "PHYSICS_FX=1: راية/عباءة بقماش حقيقي تحت ريح (بطيء على المعالج)"),
    (3, "alt", "Hair simulation", "محاكاة الشعر", ["hair_sway"], "اهتزاز شعر في المادة مرتبط بقناع الشعر، لا خصلات مُحاكاة"),
    (3, "no", "Fluids (liquid)", "محاكاة السوائل", ["water_ripple", "splash"], "لا سوائل مُحاكاة؛ تموّج ماء في المادة + رذاذ سبرايت"),
    (3, "opt", "Smoke & fire (Mantaflow)", "الدخان والنار", ["mantaflow_smoke", "mantaflow_fire", "smoke", "fire"], "PHYSICS_FX=1: Mantaflow حقيقي؛ بدونه دخان/نار سبرايت"),
    (3, "alt", "Rigid bodies", "الأجسام الصلبة", ["debris", "glass_shards"], "الحطام والشظايا حركة تحليلية حتمية (قابلة للاستئناف) لا محاكاة اصطدام"),
    (3, "alt", "Particle systems", "نظام الجسيمات", ["petals", "leaves", "snow", "embers", "dust", "rain", "fireflies", "sparks"], "غيوم رقاقات ثلاثية الأبعاد، كل رقاقة بحركتها (رفرفة الأوراق فردياً)"),
    (3, "no", "Soft body", "الأجسام الناعمة", [], "غير مستخدم"),
    (3, "alt", "Force fields", "حقول القوى", ["wind_sway", "cloth_simulation"], "حقل رياح واحد يقود الاهتزاز والجسيمات (وحقل Wind حقيقي للقماش)"),
    (3, "no", "Dynamic paint", "الرسم الديناميكي", [], "غير مستخدم"),
    (3, "no", "Geometry Nodes particles", "جسيمات عقد الهندسة", [], "غير مستخدمة (انظر الرقاقات)"),
    # ---- 5 rendering / lighting
    (4, "yes", "Cycles", "Cycles", ["engine_cycles"], "على المعالج بلا واجهة؛ GPU بـCYCLES_DEVICE=GPU"),
    (4, "opt", "EEVEE Next", "EEVEE", ["engine_eevee"], "RENDER_ENGINE=eevee؛ يحتاج OpenGL (خادم CI عادي لا يملكه)"),
    (4, "yes", "Volumetric lighting", "الإضاءة الحجمية", ["light_rays", "volume_fog", "volumetrics_principled", "mist"], "أشعة سبرايت دائماً؛ حجم Principled حقيقي بـVOLUMETRICS=1"),
    (4, "yes", "Depth of field / bokeh", "العمق الميداني", ["dof", "rack_focus"], "فتحة وبُعد بؤري حقيقيان"),
    (4, "alt", "Bloom & ambient occlusion", "التوهج والـAO", ["compositor_glare_bloom", "compositor"], "توهج: Glare في المركّب + ffmpeg. لا AO (اللوحة Emission)"),
    (4, "yes", "Global illumination", "الإضاءة العالمية", ["gi_camera_ray_trick", "lighting_key_sun"], "Cycles؛ اللوحة تُضيء نفسها فلا تتلوّن بالارتداد، والعناصر ثلاثية الأبعاد تستقبل الضوء"),
    (4, "yes", "Colour management", "إدارة الألوان (ACES)", ["color_management"], "VIEW_TRANSFORM = Standard (الافتراضي، ألوان الرسم كما هي) | AgX | Filmic | ACES"),
    (4, "opt", "Render passes / AOV", "طبقات الرندر", ["passes_exr_depth"], "RENDER_PASSES=1: Z + Mist بصيغة EXR 16-bit لكل رسمة"),
    (4, "yes", "AI denoising", "إزالة التشويش", ["denoise"], "OpenImageDenoise"),
    (4, "no", "HDRI lighting", "إضاءة HDRI", [], "الإضاءة مفتاح شمسي + ألوان اللوحة"),
    # ---- 6 editing / compositing
    (5, "alt", "Video Sequence Editor", "محطة المونتاج", [], "المونتاج في ffmpeg (assemble.py): دمج، صوت، ترجمة، نسخة عمودية"),
    (5, "yes", "Compositor nodes", "عقد الـCompositing", ["compositor", "compositor_glare_bloom"], "Glare داخل بلندر ثم تدريج ffmpeg"),
    (5, "post", "Lens distortion / chromatic aberration", "الانحراف اللوني", ["post_chroma"], "rgbashift ثابت + نبضات عند الضربات"),
    (5, "post", "Colour correction", "التصحيح اللوني", ["post_grade", "post_curves"], "عشرة تدريجات جاهزة (eq + colorbalance + curves) تُختار لكل لقطة"),
    (5, "post", "Chroma key", "مفتاح الكروما", ["post_overlay_key"], "colorkey لطبقات المؤثرات الجاهزة (شرر، تسريبات ضوء)"),
    (5, "post", "Time remapping", "تعديل السرعة", ["time_remap_len"], "hit-stop وspeed-ramp وإيقاع الرسم على اثنين"),
    (5, "post", "Film grain", "حبيبات الفيلم", ["post_grain"], "noise زمني بحسب التدريج"),
    (5, "alt", "Masks", "عزل بالأقنعة", ["rim_glow", "hair_sway"], "قناع الشخصية (rembg/عمق) يقود توهج الحافة واهتزاز الشعر؛ لا أقنعة يدوية"),
    (5, "post", "Transitions", "الانتقالات", ["post_transitions"], "cut / fade / flash / whip / dip_black / dip_white"),
    (5, "yes", "Multi-format export", "التصدير المتعدد", ["save_blend", "export_usd"], "MP4 أفقي + عمودي 9:16 (VERTICAL=1 افتراضياً)؛ .blend وUSD اختياريان لكل لقطة"),
    # ---- 7 python / automation
    (6, "yes", "Python API", "التحكم عبر بايثون", ["engine_cycles"], "كل المشهد يُبنى ببايثون من وصف اللقطة JSON"),
    (6, "yes", "Headless mode", "التشغيل بلا واجهة", ["engine_cycles"], "blender -b --factory-startup"),
    (6, "yes", "Caching / resume", "التخزين المؤقت", [], "بصمة لكل لقطة: الإطارات تُستأنف فقط إن طابقت الخطة والصور والدقة"),
    (6, "opt", "Add-ons", "الإضافات", [], "BLENDER_ADDONS=name1,name2 تُفعَّل قبل بناء المشهد (الفشل يُسجَّل ولا يوقف اللقطة)"),
    (6, "alt", "Distributed rendering", "الرندر الموزَّع", [], "مصفوفة GitHub Actions: render --shard i/n على عدة أجهزة"),
    (6, "alt", "Advanced Grease Pencil", "Grease Pencil المتقدم", ["curve_strokes"], "انظر البند 9"),
    (6, "na", "Game engine / logic bricks", "محرك الألعاب", [], "أُزيل من بلندر منذ 2.8"),
    (6, "opt", "USD & OpenVDB", "USD وOpenVDB", ["export_usd"], "EXPORT_USD=1 يكتب USD؛ OpenVDB غير مصدَّر"),
    (6, "alt", "Lazy loading", "التحميل عند الحاجة", [], "كل لقطة في عملية بلندر مستقلة تحمّل أصولها فقط"),
    (6, "yes", "Cloud-ready", "التوافق مع السحاب", [], ".github/workflows: Linux صرف، بلا واجهة، قابل للاستئناف بعد حد الـ6 ساعات"),
    # ---- 8 VFX
    (7, "yes", "Dynamic glow / bloom", "التوهج الديناميكي", ["aura", "rim_glow", "compositor_glare_bloom"], ""),
    (7, "yes", "Dolly zoom", "الدولي زوم", ["dolly_zoom"], "مطابق رياضياً بين محرك بلندر ومحرك cv"),
    (7, "yes", "Impact sparks", "شرر الضربات", ["sparks", "shockwave", "debris"], "وكذلك موجة الصدمة والحطام"),
    (7, "alt", "Force fields (wind)", "حقول الرياح", ["wind_sway"], "انظر البند 38"),
    (7, "yes", "Moving fog", "الضباب المتحرك", ["mist", "volume_fog"], ""),
    (7, "yes", "Speed lines", "خطوط السرعة", ["speed_lines"], ""),
    (7, "yes", "Heat haze", "تشويه الهواء الحراري", ["heat_haze"], ""),
    (7, "no", "Cell fracture", "التكسير الخلوي", [], "شظايا متناثرة فقط"),
    (7, "yes", "Water splashes", "رذاذ الماء", ["splash"], "رذاذ سبرايت"),
    (7, "alt", "Custom shadow shaping", "تشكيل الظل", ["toon_shading"], "عتبة الظل في تدرّج التون قابلة للضبط"),
    # ---- 9 pipeline / ecosystem
    (8, "opt", "EXR support", "دعم EXR", ["passes_exr_depth"], "الإطارات الوسيطة PNG؛ EXR للعمق/الضباب عند RENDER_PASSES=1"),
    (8, "yes", "Relative paths", "المسارات النسبية", ["save_blend"], "المشروع كله نسبي لجذره؛ ملفات .blend بمسارات نسبية"),
    (8, "yes", "Batch processing", "المعالجة الدفعية", [], "لقطة تلو لقطة، وكل لوحة تُجهَّز تسلسلياً"),
    (8, "yes", "Git / version control", "التكامل مع Git", [], "state/series.json يُلتقط في المستودع بعد كل يوم"),
    (8, "yes", "Resource tuning", "الاستهلاك الأمثل للموارد", ["motion_blur", "denoise"], "عينات متكيّفة، RENDER_SCALE للمسوّدات، محرك cv عند غياب بلندر"),
    (8, "alt", "Text & fonts", "النصوص والخطوط", ["title_3d", "title_3d_layers"], "النص يُرسم بـPillow (عربي/ياباني مشكَّل صحيحاً) ويُعرض ثلاثي الأبعاد؛ نص بلندر لا يشكّل العربية"),
    (8, "na", "Node Wrangler", "نظام العقد المرن", [], "إضافة واجهة؛ العقد تُبنى برمجياً"),
    (8, "opt", "Depth pass", "تمرير العمق", ["passes_exr_depth", "plate_depth_mesh"], "عمق اللوحة من Depth-Anything؛ تمرير Z للرندر بـRENDER_PASSES=1"),
    (8, "opt", "Multi-format asset export", "تصدير FBX/OBJ/GLTF/USD", ["export_usd"], "USD فقط"),
    (8, "na", "Updates & community", "التحديث والمجتمع", [], "غير تقني"),
    # ---- 10 cinematic polish
    (9, "yes", "Colour management profiles", "ملفات إدارة الألوان", ["color_management"], ""),
    (9, "yes", "Handheld shake", "اهتزاز الكاميرا اليدوية", ["camera_shake"], "اهتزاز + ارتجاج عند الضربة"),
    (9, "yes", "Anamorphic flares", "التوهج الأنامورفي", ["lens_flare"], "في المشهد + خطوط أفقية في ffmpeg"),
    (9, "no", "Screen-space reflections", "انعكاسات الشاشة", [], "غير مستخدمة"),
    (9, "yes", "Rack focus", "نقل التركيز", ["rack_focus"], ""),
    (9, "yes", "Motion blur", "ضبابية الحركة", ["motion_blur"], "بلندر + محرك cv + ffmpeg"),
    (9, "post", "Vignette", "الفينييت", ["post_vignette"], ""),
    (9, "post", "Contrast & saturation curves", "منحنيات التباين والتشبع", ["post_curves", "post_grade"], ""),
    (9, "alt", "Tile rendering", "الرندر المجزّأ", [], "التقسيم بحسب اللقطات على عدة مهام بدل مربعات الصورة"),
    (9, "yes", "All-in-one studio", "الاستوديو المتكامل", [], "نص/مانغا ← قصة ← صوت ← صور ← رندر ← مونتاج ← رفع، بأمر واحد"),
]

assert len(ITEMS) == 100, len(ITEMS)


def _job_features(job):
    """key -> {status: count} over the done.json of every shot of the job (and the post reports)."""
    out = collections.defaultdict(collections.Counter)
    W = work_dir(job)
    for p in sorted(W.glob("frames/shot_*/done.json")):
        try:
            feats = json.loads(p.read_text("utf-8")).get("features") or {}
        except (OSError, ValueError):
            continue
        for k, v in feats.items():
            out[k][str(v).split(":")[0]] += 1
    for p in sorted(W.glob("shots/shot_*.json")):
        try:
            feats = json.loads(p.read_text("utf-8")).get("features") or {}
        except (OSError, ValueError):
            continue
        for k, v in feats.items():
            out[k][str(v).split(":")[0]] += 1
    return out


def collect(job=None):
    seen = _job_features(job) if job else {}
    rows = []
    for cat, st, en, ar, keys, note in ITEMS:
        row = {"n": len(rows) + 1, "category": CATEGORIES[cat], "status": st, "name": en, "ar": ar, "keys": list(keys), "note": note}
        if job:
            row["seen"] = {k: dict(seen[k]) for k in keys if k in seen}
        rows.append(row)
    return rows


def report(job=None, as_json=False):
    rows = collect(job)
    counts = collections.Counter(r["status"] for r in rows)
    if as_json:
        return json.dumps({"counts": dict(counts), "items": rows}, ensure_ascii=False, indent=1)
    lines = ["الميزات المئة في بلندر وموقعها الفعلي في هذا المشروع", ""]
    cur = None
    for r in rows:
        if r["category"] != cur:
            cur = r["category"]
            lines += ["", cur]
        tail = f"  — {r['note']}" if r["note"] else ""
        seen = ""
        if r.get("seen"):
            seen = "  [" + ", ".join(f"{k}:" + "/".join(f"{s}×{n}" for s, n in v.items()) for k, v in r["seen"].items()) + "]"
        lines.append(f" {r['n']:>3}  {STATUS[r['status']]:<10} {r['ar']} ({r['name']}){tail}{seen}")
    lines += ["", "المجموع: " + " | ".join(f"{STATUS[k]} {counts[k]}" for k in STATUS),
              "ملاحظة: كود بلندر هنا مُختبَر بمحاكٍ لواجهة bpy، لا بنسخة بلندر حقيقية: شغّل `python -m mangaforge selftest` مرة على جهاز فيه بلندر."]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="features")
    ap.add_argument("--job")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    print(report(job=a.job, as_json=a.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
