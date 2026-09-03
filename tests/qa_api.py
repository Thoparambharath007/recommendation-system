"""API smoke tests against the RUNNING web server (http://127.0.0.1:8000).

Not part of unittest discovery (needs live server + loaded models).
Run: python tests/qa_api.py
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"

TEE_MARKERS = ("t shirt", "t-shirt", "tshirt", " tee ", " tees ", " t ")
TANK_MARKERS = ("tank top", " tank ", "sleeveless", "muscle", "a-shirt")
NOVELTY = ("funny", "graphic", "novelty", "gift", "st patrick", "shamrock",
           "parody", "grandma", "birthday", "christmas", "halloween", "cartoon")
COLORS = ("green", "yellow", "blue", "red", "grey", "gray", "navy",
          "black", "white", "brown", "orange", "pink", "purple")

failures: list[str] = []
checks = 0


def call(path: str, payload: dict | None = None, method: str = "POST"):
    if payload is None:
        req = urllib.request.Request(BASE + path)
    else:
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method=method,
        )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def check(name: str, cond: bool, detail: str = ""):
    global checks
    checks += 1
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(f"{name}: {detail}")


def qa_functional():
    print("\n=== FUNCTIONAL QUERIES ===")
    cases = [
        ("green shirt", "shirt"), ("white shirt for office", "shirt"),
        ("black tshirt", "tee"), ("blue polo shirt", "polo"),
        ("flannel shirt", "flannel"), ("hawaiian shirt", "hawaiian"),
        ("denim shirt", "denim"), ("pink shirt", "shirt"),
    ]
    for q, kind in cases:
        status, data = call("/api/recommend", {"query": q, "top_k": 5})
        check(f"'{q}' returns 200", status == 200, str(status))
        if status != 200:
            continue
        items = data.get("items", [])
        check(f"'{q}' elapsed sane", data.get("elapsed_s", 99) < 60, str(data.get("elapsed_s")))
        check(f"'{q}' <= 5 items", len(items) <= 5, str(len(items)))
        for it in items:
            t = it["title"].lower()
            b = (it.get("brand") or "").lower()
            t_nb = t.replace(b, "", 1) if b else t
            check(f"'{q}' item has fused_score 0..1",
                  0 <= it["fused_score"] <= 1, f"{it['fused_score']} {it['title'][:40]}")
            check(f"'{q}' item has title+asin",
                  bool(it["title"]) and bool(it["parent_asin"]))
            if kind == "shirt":
                check(f"'{q}' no tees/tanks",
                      not any(m in f" {t} " for m in TEE_MARKERS + TANK_MARKERS),
                      it["title"][:60])
            if kind == "tee":
                check(f"'{q}' only tees", any(m in f" {t} " for m in TEE_MARKERS),
                      it["title"][:60])
            for c in COLORS:
                if q.startswith(c + " ") or q.startswith(c + "shirt"):
                    check(f"'{q}' color '{c}' in title",
                          f" {c} " in f" {t_nb} ", it["title"][:60])
                    break


def qa_edge_cases():
    print("\n=== EDGE CASES ===")
    # empty query
    status, data = call("/api/recommend", {"query": "", "top_k": 5})
    check("empty query handled (200 or 422)", status in (200, 422), str(status))
    if status == 200:
        check("empty query returns list", isinstance(data.get("items"), list))
    # whitespace only
    status, _ = call("/api/recommend", {"query": "   ", "top_k": 5})
    check("whitespace query handled", status in (200, 422), str(status))
    # gibberish
    status, data = call("/api/recommend", {"query": "xzqjkw vulfmonic", "top_k": 5})
    check("gibberish query returns 200", status == 200, str(status))
    if status == 200:
        check("gibberish still returns items", len(data["items"]) > 0)
    # special characters
    status, _ = call("/api/recommend", {"query": "shirt!!! @#$% <>\"'", "top_k": 5})
    check("special chars handled", status == 200, str(status))
    # emoji
    status, _ = call("/api/recommend", {"query": "blue shirt 😎", "top_k": 5})
    check("emoji query handled", status == 200, str(status))
    # very long query
    status, _ = call("/api/recommend", {"query": "comfortable cotton shirt " * 100, "top_k": 5})
    check("long query handled", status == 200, str(status))
    # top_k extremes
    status, data = call("/api/recommend", {"query": "white shirt", "top_k": 1})
    check("top_k=1 works", status == 200 and len(data.get("items", [])) <= 1, str(status))
    status, data = call("/api/recommend", {"query": "white shirt", "top_k": 50})
    check("top_k=50 works", status == 200, str(status))
    # invalid top_k
    status, _ = call("/api/recommend", {"query": "white shirt", "top_k": 0})
    check("top_k=0 rejected/handled", status in (200, 422), str(status))
    status, _ = call("/api/recommend", {"query": "white shirt", "top_k": -5})
    check("negative top_k rejected", status == 422, str(status))
    status, _ = call("/api/recommend", {"query": "white shirt", "top_k": 10_000})
    check("huge top_k rejected/handled", status in (200, 422), str(status))
    # weight extremes
    status, data = call("/api/recommend", {
        "query": "linen shirt", "top_k": 5,
        "w_content": 1.0, "w_embed": 0.0, "w_sentiment": 0.0})
    check("pure content weights work", status == 200, str(status))
    status, data = call("/api/recommend", {
        "query": "linen shirt", "top_k": 5,
        "w_content": 0.0, "w_embed": 1.0, "w_sentiment": 0.0})
    check("pure embed weights work", status == 200, str(status))
    status, _ = call("/api/recommend", {
        "query": "linen shirt", "top_k": 5,
        "w_content": 999, "w_embed": 0, "w_sentiment": 0})
    check("out-of-range weights rejected", status == 422, str(status))
    # min_reviews
    status, data = call("/api/recommend", {"query": "white shirt", "top_k": 5, "min_reviews": 20})
    check("min_reviews=20 works", status == 200, str(status))
    # malformed JSON
    try:
        req = urllib.request.Request(
            BASE + "/api/recommend", data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    check("malformed JSON -> 422", code == 422, str(code))


def qa_data_quality():
    print("\n=== DATA QUALITY (sampled) ===")
    status, data = call("/api/recommend", {"query": "cotton dress shirt", "top_k": 20})
    if status != 200:
        check("data quality probe", False, f"status {status}")
        return
    items = data["items"]
    n_price = sum(1 for i in items if i.get("price") is not None)
    n_img = sum(1 for i in items if i.get("image_url"))
    n_brand = sum(1 for i in items if i.get("brand"))
    bad_enc = [i["title"] for i in items
               if any(s in i["title"] for s in ("Ã¢", "Ã©", "Â®", "â€"))]
    check(">= half items have price", n_price >= len(items) // 2, f"{n_price}/{len(items)}")
    check("most items have image", n_img >= len(items) * 0.8, f"{n_img}/{len(items)}")
    check("no mojibake in titles", not bad_enc, "; ".join(bad_enc[:3]))
    print(f"       sample: price {n_price}/{len(items)}, img {n_img}/{len(items)}, brand {n_brand}/{len(items)}")


def main():
    h_status, h = call("/health", method="GET")
    check("server healthy", h_status == 200 and h.get("ready") is True, str(h))
    if failures:
        pass
    qa_functional()
    qa_edge_cases()
    qa_data_quality()
    print(f"\n{'=' * 50}\nTOTAL: {checks} checks, {len(failures)} failures")
    for f in failures:
        print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
