"""Validate papers.json against its own meta (fields, taxonomy, vocabularies) and the local files.

    python research/check_papers.py

Errors (exit code 1): duplicate or malformed ids, ids that do not match their file stem, taxonomy values
outside meta.taxonomy, categorical values outside their vocabulary, missing files, PDFs without Markdown,
missing or unknown fields, wrong types, duplicates inside arrays, country lists that do not match the
countries of the listed institutions, malformed vocabulary entries, malformed `code` items (keys, URL, licence,
commit hash, local path pattern) and local code clones that are missing.
Warnings: vocabulary entries or taxonomy strategies never used, entries without affiliations,
open-access PDFs without a local copy and no note, unindexed files in papers-original/, code clones whose HEAD differs
from the recorded commit, clones in code/ that neither papers.json nor code/README.md lists.
Prints a summary (papers per L1, L2, country, institution type, year; code links) at the end.

The clones in code/ are only read as text (.git/HEAD and refs); nothing in them is imported or executed.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPERS, MARKDOWN, CODE = HERE / "papers-original", HERE / "papers-md", HERE / "code"
INSTITUTION_TYPES = {"education", "company", "government", "facility", "nonprofit", "healthcare", "archive", "other"}
OPTIONAL = {"kind", "code"}
ARRAYS = ("contribution", "input", "country", "institution")
CODE_KEYS = {"url", "licence", "commit", "local"}

errors, warnings = [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def clone_head(path):
    """Commit hash of a clone's HEAD, read from .git/HEAD and the refs (no git call); None if unreadable."""
    git = path / ".git"
    if not git.is_dir():
        return None
    head = (git / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    if not head.startswith("ref: "):
        return head if re.fullmatch(r"[0-9a-f]{40}", head) else None
    ref = head[5:].strip()
    loose = git / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8", errors="replace").strip()
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def check_code(where, code):
    """`code`: non-empty list of {url, licence, commit, local}; local, when set, is an existing clone in research/code/."""
    if not isinstance(code, list) or not code:
        err(f"{where} code must be a non-empty array of {{url, licence, commit, local}}")
        return
    urls = Counter()
    for item in code:
        if not isinstance(item, dict) or set(item) != CODE_KEYS:
            err(f"{where} code item needs exactly {sorted(CODE_KEYS)}: {item!r}")
            continue
        url, licence, commit, local = item["url"], item["licence"], item["commit"], item["local"]
        urls[url] += 1
        if not isinstance(url, str) or not re.fullmatch(r"https://[^\s/]+/\S+", url):
            err(f"{where} code url {url!r} must be an https URL")
            continue
        if not isinstance(licence, str) or not licence.strip():
            err(f"{where} code {url}: licence must be a non-empty string ('none' when the repository has no licence file)")
        if commit is not None and not (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)):
            err(f"{where} code {url}: commit must be a full 40-character hash or null")
            continue
        if local is None:
            continue
        if not isinstance(local, str) or not re.fullmatch(r"research/code/[\w.-]+__[\w.-]+", local):
            err(f"{where} code {url}: local {local!r} must be research/code/<owner>__<repo> or null")
            continue
        m = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?", url)
        if m and local != f"research/code/{m[1]}__{m[2]}":
            err(f"{where} code {url}: local {local!r} does not match the repository (research/code/{m[1]}__{m[2]})")
        path = ROOT / local
        if not path.is_dir():
            err(f"{where} code {url}: local clone {local} missing (see research/code/README.md)")
            continue
        if commit is None:
            err(f"{where} code {url}: commit is null although a local clone is set")
            continue
        head = clone_head(path)
        if head is None:
            warn(f"{where} code {local}: cannot read the clone's HEAD")
        elif head != commit:
            warn(f"{where} code {local}: clone is at {head[:12]}, papers.json records {commit[:12]}")
    for url, n in urls.items():
        if n > 1:
            err(f"{where} code lists {url} {n} times")


def check(data):
    meta, entries = data["meta"], data["papers"]
    fields, taxonomy, vocab = meta["fields"], meta["taxonomy"], meta["vocabularies"]
    countries, institutions = vocab["country"], vocab["institution"]
    used = {name: Counter() for name in ("contribution", "input", "domain", "country", "institution", "affiliation_source")}
    used_l2 = Counter()

    # vocabularies themselves
    for code, name in countries.items():
        if not re.fullmatch(r"[A-Z]{2}", code) or not isinstance(name, str) or not name:
            err(f"vocabularies.country: bad entry {code!r}: {name!r}")
    rors = Counter()
    for name, spec in institutions.items():
        if not isinstance(spec, dict) or set(spec) != {"country", "type", "ror"}:
            err(f"vocabularies.institution[{name!r}]: needs exactly country, type, ror")
            continue
        if spec["country"] is not None and spec["country"] not in countries:
            err(f"vocabularies.institution[{name!r}]: country {spec['country']!r} not in vocabularies.country")
        if spec["type"] not in INSTITUTION_TYPES:
            err(f"vocabularies.institution[{name!r}]: type {spec['type']!r} not one of {sorted(INSTITUTION_TYPES)}")
        if spec["ror"] is not None:
            if not re.fullmatch(r"https://ror\.org/0[a-z0-9]{8}", spec["ror"]):
                err(f"vocabularies.institution[{name!r}]: malformed ror {spec['ror']!r}")
            rors[spec["ror"]] += 1
    for ror, n in rors.items():
        if n > 1:
            err(f"vocabularies.institution: {ror} used by {n} names (one spelling per institution)")
    folded = Counter(re.sub(r"\W", "", n.casefold()) for n in institutions)
    for name in institutions:
        if folded[re.sub(r"\W", "", name.casefold())] > 1:
            err(f"vocabularies.institution: {name!r} differs from another name only in case or punctuation")

    ids = Counter(e.get("id") for e in entries)
    for i, n in ids.items():
        if n > 1:
            err(f"id {i!r} used {n} times")

    for e in entries:
        pid = e.get("id", "<no id>")
        where = f"{pid}:"
        missing = [f for f in fields if f not in OPTIONAL and f not in e]
        unknown = [f for f in e if f not in fields]
        if missing:
            err(f"{where} missing fields {missing}")
        if unknown:
            err(f"{where} unknown fields {unknown} (document them in meta.fields)")
        if not re.fullmatch(r"\d{4}-[a-z0-9]+(-[a-z0-9]+)+", str(pid)):
            err(f"{where} id must be year-surname-topic (lowercase, hyphens)")
        if not isinstance(e.get("year"), int):
            err(f"{where} year must be an integer")
        elif not str(pid).startswith(str(e["year"])):
            warn(f"{where} id does not start with year {e['year']}")
        for f in ("title", "authors", "venue", "page"):
            if not isinstance(e.get(f), str) or not e.get(f):
                err(f"{where} {f} must be a non-empty string")
        if "kind" in e and e["kind"] != "web":
            err(f"{where} kind must be 'web' when present")

        # taxonomy
        t = e.get("taxonomy") or {}
        if t.get("l1") not in taxonomy:
            err(f"{where} taxonomy.l1 {t.get('l1')!r} not in meta.taxonomy")
        elif t.get("l2") not in taxonomy[t["l1"]]["l2"]:
            err(f"{where} taxonomy.l2 {t.get('l2')!r} not a strategy of {t['l1']!r}")
        else:
            used_l2[(t["l1"], t["l2"])] += 1

        # categorical arrays
        for f in ARRAYS:
            v = e.get(f)
            if not isinstance(v, list):
                err(f"{where} {f} must be an array")
                continue
            dup = [x for x, n in Counter(v).items() if n > 1]
            if dup:
                err(f"{where} {f} has duplicates {dup}")
            for x in v:
                if x not in vocab[f]:
                    err(f"{where} {f} value {x!r} not in vocabularies.{f}")
                used[f][x] += 1
        for f in ("contribution", "input"):
            if isinstance(e.get(f), list) and not e[f]:
                err(f"{where} {f} is empty")
        for f in ("domain", "affiliation_source"):
            v = e.get(f)
            if v not in vocab[f]:
                err(f"{where} {f} {v!r} not in vocabularies.{f}")
            else:
                used[f][v] += 1

        # affiliations: country must be the countries of the institutions, in order of first appearance
        inst, ctry = e.get("institution") or [], e.get("country") or []
        if isinstance(inst, list) and isinstance(ctry, list) and all(i in institutions for i in inst):
            derived = list(dict.fromkeys(institutions[i]["country"] for i in inst if institutions[i]["country"]))
            if set(derived) != set(ctry):
                err(f"{where} country {ctry} does not match the institutions' countries {derived}")
            elif derived != ctry:
                warn(f"{where} country order {ctry} differs from institution order {derived}")
        if not inst:
            warn(f"{where} no institution (affiliations not determined)")

        # files
        file, pdf = e.get("file"), e.get("pdf")
        if e.get("kind") == "web":
            if file and not (HERE / file).exists():
                err(f"{where} saved page {file!r} missing")
        elif file:
            if Path(file).stem != pid:
                err(f"{where} file stem {Path(file).stem!r} does not match id")
            if not (PAPERS / file).exists():
                err(f"{where} file papers-original/{file} missing")
            elif file.endswith(".pdf") and not (MARKDOWN / f"{Path(file).stem}.md").exists():
                err(f"{where} no Markdown papers-md/{Path(file).stem}.md (run convert_papers.py)")
        elif pdf and "download blocked" not in (e.get("note") or "").lower():
            warn(f"{where} open-access PDF but no local file and no 'download blocked' note")

        # code repositories
        if "code" in e:
            check_code(where, e["code"])

    # code clones on disk that nothing lists
    listed = {c.get("local") for e in entries for c in (e.get("code") or []) if isinstance(c, dict)}
    readme = (CODE / "README.md").read_text(encoding="utf-8") if (CODE / "README.md").is_file() else ""
    for d in sorted(p for p in CODE.iterdir() if p.is_dir()) if CODE.exists() else []:
        slug = d.name.replace("__", "/", 1)                  # GitHub owners cannot contain underscores
        if f"research/code/{d.name}" not in listed and f"github.com/{slug})" not in readme:
            warn(f"research/code/{d.name}: clone listed neither in papers.json nor in code/README.md")

    # files on disk
    indexed = {e.get("file") for e in entries}
    for p in sorted(PAPERS.glob("*.pdf")) if PAPERS.exists() else []:
        if not (MARKDOWN / f"{p.stem}.md").exists():
            err(f"papers-original/{p.name}: no Markdown in papers-md/")
        if p.name not in indexed:
            warn(f"papers-original/{p.name}: not in papers.json")

    # unused vocabulary entries and strategies
    for f, counter in used.items():
        for value in vocab[f]:
            if not counter[value]:
                warn(f"vocabularies.{f}: {value!r} never used")
    for l1, spec in taxonomy.items():
        for l2 in spec["l2"]:
            if not used_l2[(l1, l2)]:
                warn(f"taxonomy: {l1} / {l2} has no papers")
    return used, used_l2


def summary(data, used, used_l2):
    entries, vocab = data["papers"], data["meta"]["vocabularies"]
    print(f"\n{len(entries)} entries")

    def table(title, counter, label=str):
        print(f"\n{title}")
        for key, n in counter:
            print(f"  {n:>3}  {label(key)}")

    l1 = Counter(e["taxonomy"]["l1"] for e in entries)
    table("Papers per approach family / contribution type (L1)", l1.most_common())
    table("Papers per strategy (L1 / L2)", sorted(used_l2.items(), key=lambda kv: (-kv[1], kv[0])), lambda k: f"{k[0]} / {k[1]}")
    table("Papers per country (a paper counts once per country)", used["country"].most_common(),
          lambda c: f"{c} {vocab['country'].get(c, '?')}")
    types = Counter(t for e in entries for t in {vocab["institution"][i]["type"] for i in e["institution"] if i in vocab["institution"]})
    table("Papers per institution type (a paper counts once per type)", types.most_common())
    table("Papers per year", sorted(Counter(e["year"] for e in entries).items()))
    print(f"\n{len(vocab['institution'])} institutions, {len(vocab['country'])} countries in the vocabularies")
    links = [c for e in entries for c in (e.get("code") or []) if isinstance(c, dict)]
    print(f"{sum(1 for e in entries if e.get('code'))} entries with code: {len(links)} repository links "
          f"({len({c.get('url') for c in links})} repositories), {sum(1 for c in links if c.get('local'))} with a local clone, "
          f"{sum(1 for c in links if str(c.get('licence', '')).startswith('none'))} without a licence file")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads((HERE / "papers.json").read_text(encoding="utf-8"))
    used, used_l2 = check(data)
    for w in warnings:
        print(f"WARNING {w}")
    for x in errors:
        print(f"ERROR   {x}")
    summary(data, used, used_l2)
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)
