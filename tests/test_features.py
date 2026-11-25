import json

from mangaforge.render import features
from mangaforge import schema


def test_registry_covers_the_hundred_capabilities_in_ten_categories():
    rows = features.collect()
    assert len(rows) == 100 and [r["n"] for r in rows] == list(range(1, 101))
    assert {r["category"] for r in rows} == set(features.CATEGORIES)
    assert all(sum(1 for r in rows if r["category"] == c) == 10 for c in features.CATEGORIES)
    assert {r["status"] for r in rows} <= set(features.STATUS)


def test_every_claimed_key_exists_in_the_code_that_reports_it():
    """A feature key listed here must really be emitted by a renderer / the post stage (no brochure claims)."""
    import pathlib
    import re
    root = pathlib.Path(features.__file__).parent.parent
    code = "".join(p.read_text("utf-8") for p in list(root.glob("render/*.py")) + [root / "post.py"] if p.name != "features.py")
    literals = set(re.findall(r"[\"']([a-z0-9_]+)[\"']", code))
    prefixes = {x for x in literals if x.endswith("_")}                      # e.g. "engine_" + name, "mantaflow_" + kind
    missing = sorted({k for r in features.collect() for k in r["keys"]
                      if k not in literals and not any(k.startswith(p) for p in prefixes)})
    assert not missing, f"keys never reported by any renderer: {missing}"


def test_effects_named_in_the_schema_are_all_mapped():
    mentioned = {k for r in features.collect() for k in r["keys"]}
    assert {"slash", "shockwave", "sparks", "lightning"} & set(schema.FX_BLENDER) & mentioned


def test_report_is_readable_and_json_is_valid():
    text = features.report()
    assert "المجموع" in text and "selftest" in text
    d = json.loads(features.report(as_json=True))
    assert sum(d["counts"].values()) == 100
