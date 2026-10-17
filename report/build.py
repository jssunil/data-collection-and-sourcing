"""Build the Netlify site in site/ from report/template.html, the results JSON files and the docs.

The charts and the per-language table read the injected DATA object, so their numbers come
straight from results/. The Markdown documents become linked pages with the same style.
Run after all experiments:  python report/build.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_json  # noqa: E402

SITE = ROOT / "site"
REPO = "https://github.com/jssunil/data-collection-and-sourcing"
# (source, page, nav label, page title)
DOCS = [
    ("PLAN.md", "plan.html", "Plan", "Experiment Plan"),
    ("SESSION_03_EXPLAINER.md", "explainer.html", "Explainer", "Session 3 Explainer"),
    ("FIRST_PRINCIPLES_QA.md", "questions.html", "First principles", "First-Principles Questions"),
    ("README.md", "evidence.html", "Evidence", "Evidence and Results"),
    ("assignment.md", "assignment.html", "Assignment", "Assignment Brief"),
]
PAGE = {src: page for src, page, _, _ in DOCS} | {"report/index.html": "index.html"}

DOC_CSS = """
nav.top{max-width:880px;margin:0 auto;padding:14px 16px 0;display:flex;flex-wrap:wrap;gap:6px 16px;font-size:14px}
nav.top a{text-decoration:none;color:var(--ink2)}
nav.top a.on{color:var(--ink);font-weight:700}
main.doc pre{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px;overflow-x:auto}
main.doc pre code{font-size:12.5px}
main.doc blockquote{margin:10px 0;padding:2px 14px;border-left:3px solid var(--grid);color:var(--ink2)}
main.doc h1{margin-top:8px}
main.doc hr{display:none}
main.doc img{max-width:100%;height:auto;background:#fff;border-radius:8px}
"""


def nav(current):
    items = [("index.html", "Report")] + [(p, label) for _, p, label, _ in DOCS] + [(REPO, "Code")]
    links = "".join(f'<a href="{h}"{" class=on" if h == current else ""}>{t}</a>' for h, t in items)
    return f'<nav class="top">{links}</nav>'


def fix_link(m):
    href = m.group(1)
    if re.match(r"[a-z]+:|#", href):
        return m.group(0)
    path, _, frag = href.partition("#")
    if path in PAGE:
        new = PAGE[path]
    elif path.startswith("results/") and path.endswith(".png"):
        new = path
    else:
        new = f"{REPO}/blob/main/{path}"
    return f'href="{new}{"#" + frag if frag else ""}"'


def build_report(style):
    s = load_json("summary")
    e2 = load_json("e2_vocab_sweep")
    e3 = load_json("e3_cleaning")
    data = {
        "sweep": {mix: [{"V": r["V"], "t1": r["mean_fertility_tier1"], "t2": r["mean_fertility_tier2"],
                         "eng": r["lang"]["eng"]["fertility"], "code": r["domain"]["code_github"]["bytes_per_token"]}
                        for r in e2[mix]["rows"]] for mix in ("india", "english") if mix in e2},
        "cost": [{"V": c["V"], "rel": c["rel_cost"]} for c in e2["india"]["cost"]],
        "ref": {n: s["tier_means"][n][0] for n in ("gemma-4", "sarvam-30b", "brahmic-131k", "gpt-oss")},
        "cleaning": [{"lang": c, "en": e3[c]["english"]["any"], "aware": e3[c]["aware"]["any"]} for c in e3],
        "langs": s["per_language"],
        "domains": s["domains"],
    }
    html = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    html = html.replace("</style>", DOC_CSS + "</style>", 1).replace("<main>", nav("index.html") + "\n<main>", 1)
    (SITE / "index.html").write_text(html, encoding="utf-8")


def build_doc(src, page, title, head):
    text = (ROOT / src).read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = re.sub(r'href="([^"]*)"', fix_link, body)
    body = body.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")
    html = (f"{head}<title>{title}</title>\n<style>{DOC_CSS}</style>\n</head>\n<body>\n{nav(page)}\n"
            f'<main class="doc">\n{body}\n<p class="muted">Source: <a href="{REPO}/blob/main/{src}">{src}</a></p>\n'
            "</main>\n</body>\n</html>\n")
    (SITE / page).write_text(html, encoding="utf-8")


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "results").mkdir(parents=True)
    template = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
    style = template[template.index("<style>"):template.index("</style>") + len("</style>")]
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + style + "\n")
    build_report(style)
    for src, page, _, title in DOCS:
        build_doc(src, page, title, head)
    for png in (ROOT / "results").glob("*.png"):
        shutil.copy(png, SITE / "results" / png.name)
    files = sorted(p.relative_to(SITE).as_posix() for p in SITE.rglob("*") if p.is_file())
    print("wrote site/:", ", ".join(files), f"({sum((SITE / f).stat().st_size for f in files) // 1024} KB)")


if __name__ == "__main__":
    main()
