"""Download the open-access papers listed in papers.json into papers-original/ and write README.md.

papers.json holds {"meta": ..., "papers": [...]}: meta documents the fields, the two-level taxonomy and the controlled
vocabularies (contribution, input, domain, country, institution); research/README.md lists the papers by taxonomy.

    python research/fetch_papers.py            # download missing PDFs, then rebuild README.md
    python research/fetch_papers.py --index    # only rebuild README.md

PDFs are for private reading and are gitignored: most arXiv papers grant arXiv only a distribution
licence, so they must not be republished. Uses curl (ships with Windows 10+, macOS and Linux).
Waits between requests to stay within arXiv's guidance for automated access.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE / "papers-original"
MARKDOWN = HERE / "papers-md"
UA = "Mozilla/5.0 (floorplan-extract research index)"


def download(entry):
    target = PAPERS / entry["file"]
    if target.exists() and target.read_bytes()[:4] == b"%PDF":
        return "present"
    tmp = target.with_suffix(".part")
    result = subprocess.run(["curl", "-fsSL", "--max-time", "180", "-A", UA, "-o", str(tmp), entry["pdf"]],
                            capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.read_bytes()[:4] != b"%PDF":
        tmp.unlink(missing_ok=True)
        return f"failed ({result.stderr.strip()[:80] or 'not a PDF'})"
    tmp.replace(target)
    return "downloaded"


def load():
    return json.loads((HERE / "papers.json").read_text(encoding="utf-8"))


def write_index(data):
    meta, entries = data["meta"], data["papers"]
    lines = [
        "# Research",
        "",
        "- Literature review and state of the art: [docs/literature-review.md](../docs/literature-review.md) – methods, models, benchmarks, datasets, findings by pipeline stage",
        "- Sources: [sources.md](sources.md) – the numbered source list shared by all documents",
        "- Motivation and goals: [docs/motivation-goals.md](../docs/motivation-goals.md)",
        "- Ready-to-use solutions: [docs/solutions-closed.md](../docs/solutions-closed.md) (closed), [docs/solutions-open.md](../docs/solutions-open.md) (open)",
        "- Pipeline design: [docs/pipeline.md](../docs/pipeline.md)",
        "",
        "## Papers and sources",
        "",
        f"{len(entries)} papers and practitioner sources referenced in the documents and the pilots, listed by the taxonomy "
        "in [papers.json](papers.json): approach family or contribution type (sections), then strategy. "
        "`meta` in papers.json defines every field and vocabulary (taxonomy, contribution, input, domain, country, institution).",
        "",
        "| Folder | Content |",
        "|---|---|",
        "| `papers-incoming/` | New, unsorted PDFs. Name them `year-author-topic.pdf`, add them to `papers.json`, move them to `papers-original/` |",
        "| `papers-original/` | Indexed PDFs; `python research/fetch_papers.py` downloads the open-access ones |",
        "| `papers-md/` | Markdown versions; `python research/convert_papers.py` converts new PDFs with Docling |",
        "| `web/` | Saved web pages |",
        "| `code/` | Shallow clones of the papers' public code and of pipeline components, for reading only (never install or run them); "
        "[code/README.md](code/README.md) lists them with licence and commit, and `code` in `papers.json` links them to the papers |",
        "",
        "These folders are gitignored (except `code/README.md`): the files are for private reading and must not be republished. "
        "`fetch_papers.py --index` rebuilds this README from `papers.json`.",
        "",
    ]
    for l1, spec in meta["taxonomy"].items():
        group = [e for e in entries if e["taxonomy"]["l1"] == l1]
        if not group:
            continue
        order = list(spec["l2"])
        lines += [f"### {l1}", "", spec["description"], "",
                  "| Year | Paper | Strategy | Venue | Local copy | Note |", "|---|---|---|---|---|---|"]
        for e in sorted(group, key=lambda e: (order.index(e["taxonomy"]["l2"]), e["year"], e["id"])):
            if e.get("kind") == "web":
                local = f"[saved page](<{e['file']}>)" if e["file"] and (HERE / e["file"]).exists() else "web page"
            elif e["file"] and (PAPERS / e["file"]).exists():
                md = Path(e["file"]).with_suffix(".md").name
                local = f"[PDF](papers-original/{e['file']})" + (f" · [MD](papers-md/{md})" if (MARKDOWN / md).exists() else "")
            else:
                local = "not open access" if not e["pdf"] else "not downloaded"
            lines.append(f"| {e['year']} | [{e['title']}]({e['page']}) – {e['authors']} | {e['taxonomy']['l2']} | {e['venue']} | {local} | {e['note']} |")
        lines.append("")
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    data = load()
    entries = data["papers"]
    if "--index" not in sys.argv:
        PAPERS.mkdir(exist_ok=True)
        for e in entries:
            if not e["pdf"]:
                if e.get("kind") != "web":
                    print(f"skip (not open access)  {e['title'][:70]}")
                continue
            if not e["file"]:                           # open access, but the publisher blocks automated downloads
                print(f"skip (download blocked) {e['title'][:70]}")
                continue
            status = download(e)
            print(f"{status:<12} {e['file']}")
            if status == "downloaded":
                time.sleep(3.5)
    write_index(data)
    print("README.md written")
