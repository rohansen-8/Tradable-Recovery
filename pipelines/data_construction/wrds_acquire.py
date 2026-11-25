"""Download the raw WRDS data for the option-panel reconstruction, with a manifest.

The queries are those of the authors' Python data pipeline, OptPortOpt_V2
(data/cleaning/wrds_pull.py and wrds_pull_spx_vix_zerocd.py): same libraries,
tables, fields, secid filter and year range. Each query template below is
asserted to occur verbatim in that source, so any divergence fails loudly.
Execution details, none of which changes a query:
  * output root is <repo>/data_reconstructed/raw/wrds;
  * the WRDS username comes from $WRDS_USER; the password is read by libpq
    from ~/.pgpass, never here;
  * options/forwards are appended to the CSV year by year (same columns,
    same running index column, same pandas to_csv defaults). An integer column that is integral in one
    year and NaN-bearing in another prints as "5" vs "5.0" across years; this
    changes text, not parsed values;
  * an existing target file is never overwritten (the run stops instead).

Modes:
  --dry-run   print the exact queries and targets; no connection.
  --count     connect and run COUNT/MIN/MAX per query (no rows downloaded).
  --pull      download, then write MANIFEST.json (rows, dates, size, SHA-256).
Select datasets with --only options,forwards,zerocd,index (default: all).
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

# Pull scripts of the authors' Python data pipeline (OptPortOpt_V2/data/cleaning); set OPTPORTOPT_V2 to that repository root.
COAUTHOR = Path(os.environ.get("OPTPORTOPT_V2", "../OptPortOpt_V2")).expanduser() / "data" / "cleaning"
OUT = Path(__file__).resolve().parents[2] / "data_reconstructed" / "raw" / "wrds"


def coauthor_constants():
    """Module-level literals of the two pull scripts, read without importing them."""
    out = {}
    for name in ("wrds_pull.py", "wrds_pull_spx_vix_zerocd.py"):
        src = (COAUTHOR / name).read_text()
        consts = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                try:
                    consts[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        out[name] = (src, consts)
    return out


def build_jobs():
    """Exact query list of the OptPortOpt_V2 pull scripts, each template checked against the source."""
    C = coauthor_constants()
    src1, k1 = C["wrds_pull.py"]
    src2, k2 = C["wrds_pull_spx_vix_zerocd.py"]
    for tmpl, src in (
            ('f"SELECT {select} FROM {table_fmt.format(year=year)} "', src1),
            ('f"WHERE secid = {SECID}"', src1),
            ('"optionm_all.opprcd{year}", OPTION_FIELDS', src1),
            ('"optionm_all.fwdprd{year}", "*", "forwards"', src1),
            ('f"SELECT date, days, rate FROM optionm_all.zerocd "', src2),
            ("f\"WHERE date >= '{START_YEAR}-01-01' \"", src2),
            ("f\"AND date <= '{END_YEAR}-12-31'\"", src2),
            ('"SELECT secid, ticker, issuer FROM optionm_all.securd1 "', src2),
            ('"WHERE ticker = \'VIX\'"', src2),
            ('f"SELECT date, open, high, low, close, volume "', src2),
            ('f"FROM optionm_all.secprd{year} WHERE secid = {sid}"', src2)):
        assert tmpl in src, f"Query template not found in co-author source: {tmpl}"
    assert k1["SECID"] == k2["SECID_SPX"] == 108105
    y1, y2 = range(k1["START_YEAR"], k1["END_YEAR"] + 1), range(k2["START_YEAR"], k2["END_YEAR"] + 1)
    yearly = lambda tbl, sel, sid: [(y, f"SELECT {sel} FROM optionm_all.{tbl}{y} WHERE secid = {sid}") for y in y1]
    return {
        "options": dict(file="df_wrds_options.csv", index=True, sort=None, source="wrds_pull.py",
                        parts=yearly("opprcd", k1["OPTION_FIELDS"], k1["SECID"])),
        "forwards": dict(file="df_wrds_forwards.csv", index=True, sort=None, source="wrds_pull.py",
                         parts=yearly("fwdprd", "*", k1["SECID"])),
        "zerocd": dict(file="df_wrds_zerocd.csv", index=False, sort=["date", "days"],
                       source="wrds_pull_spx_vix_zerocd.py",
                       parts=[(None, "SELECT date, days, rate FROM optionm_all.zerocd "
                                     f"WHERE date >= '{y2.start}-01-01' AND date <= '{y2.stop - 1}-12-31'")]),
        "index": dict(source="wrds_pull_spx_vix_zerocd.py", years=y2, spx=k2["SECID_SPX"],
                      lookup="SELECT secid, ticker, issuer FROM optionm_all.securd1 WHERE ticker = 'VIX'"),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def count_sql(q):
    """COUNT/MIN/MAX(date) version of a SELECT ... FROM ... query."""
    return "SELECT COUNT(*) AS n, MIN(date) AS dmin, MAX(date) AS dmax FROM" + q.split(" FROM", 1)[1]


def pull_to_csv(db, name, job, target, record):
    """Download one dataset part by part into target; record per-part rows and dtypes."""
    import pandas as pd
    assert not target.exists(), f"{target} exists; refusing to overwrite"
    tmp = target.with_suffix(target.suffix + ".part")
    assert not tmp.exists(), f"{tmp} exists; inspect and remove it first"
    offset, parts = 0, []
    if job["sort"]:  # single small query, sorted as in wrds_pull_spx_vix_zerocd.py
        y, q = job["parts"][0]
        df = db.raw_sql(q)
        assert len(df) > 0, f"Empty {name} pull"
        df = df.sort_values(job["sort"]).reset_index(drop=True)
        df.to_csv(tmp, index=job["index"])
        parts.append(dict(year=y, query=q, rows=len(df), dtypes={c: str(t) for c, t in df.dtypes.items()}))
        offset = len(df)
    else:
        for i, (y, q) in enumerate(job["parts"]):
            df = db.raw_sql(q)
            assert len(df) > 0, f"Empty {name} pull for {y}"
            df.index = range(offset, offset + len(df))
            df.to_csv(tmp, mode="a", header=(i == 0), index=job["index"])
            parts.append(dict(year=y, query=q, rows=len(df), date_min=str(df["date"].min()),
                              date_max=str(df["date"].max()), dtypes={c: str(t) for c, t in df.dtypes.items()}))
            offset += len(df)
            print(f"  {name} {y}: {len(df)} rows", flush=True)
    os.replace(tmp, target)
    record_file(record, name, job["source"], target, parts, offset)


def record_file(record, name, source, target, parts, rows):
    import pandas as pd
    d = pd.read_csv(target, usecols=["date"])["date"]
    record[target.name] = dict(dataset=name, coauthor_script=source, rows=rows,
                               date_min=str(d.min()), date_max=str(d.max()),
                               bytes=target.stat().st_size, sha256=sha256(target),
                               downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               parts=parts)


def pull_index(db, job, record):
    """SPX and VIX OHLC from secprd; VIX secid resolved from securd1 as in wrds_pull_spx_vix_zerocd.py."""
    import pandas as pd
    look = db.raw_sql(job["lookup"])
    assert len(look) >= 1, "VIX not found in optionm_all.securd1"
    look.to_csv(OUT / "securd1_vix_lookup.csv", index=False)  # all matches kept for provenance
    vix_secid = int(look.iloc[0]["secid"])  # first match, as in the source script
    print(f"  VIX secid resolved: {vix_secid} ({len(look)} securd1 rows)")
    for label, sid in (("spx", job["spx"]), ("vix", vix_secid)):
        target = OUT / f"df_wrds_{label}_prices.csv"
        assert not target.exists(), f"{target} exists; refusing to overwrite"
        parts, chunks = [], []
        for y in job["years"]:
            q = f"SELECT date, open, high, low, close, volume FROM optionm_all.secprd{y} WHERE secid = {sid}"
            c = db.raw_sql(q)
            assert len(c) > 0, f"Empty secprd pull for {label} {y}"
            chunks.append(c)
            parts.append(dict(year=y, query=q, rows=len(c)))
        df = pd.concat(chunks, ignore_index=True).sort_values("date").reset_index(drop=True)
        df.to_csv(target, index=False)
        record_file(record, f"index_{label}", job["source"], target, parts, len(df))
        record[target.name]["secid"] = sid


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--pull", action="store_true")
    ap.add_argument("--only", default="options,forwards,zerocd,index")
    a = ap.parse_args()
    jobs = build_jobs()
    sel = [s.strip() for s in a.only.split(",")]
    assert set(sel) <= set(jobs), f"--only must be a subset of {sorted(jobs)}"

    if a.dry_run:
        for s in sel:
            j = jobs[s]
            if s == "index":
                print(f"[index] {j['lookup']}")
                print(f"[index] SELECT date, open, high, low, close, volume FROM optionm_all.secprd<{j['years'].start}..{j['years'].stop - 1}> WHERE secid = <108105 | VIX>")
                continue
            print(f"[{s}] -> {OUT / j['file']}  ({len(j['parts'])} queries)")
            for y, q in (j["parts"][:1] + j["parts"][-1:] if len(j["parts"]) > 1 else j["parts"]):
                print(f"   {q}")
        return

    import wrds
    user = os.environ.get("WRDS_USER")
    assert user, "Set WRDS_USER in your shell (the password stays in ~/.pgpass)."
    db = wrds.Connection(wrds_username=user)
    try:
        if a.count:
            for s in sel:
                j = jobs[s]
                qs = j["parts"] if s != "index" else \
                    [(y, f"SELECT date FROM optionm_all.secprd{y} WHERE secid = {j['spx']}") for y in j["years"]]
                tot = 0
                for y, q in qs:
                    r = db.raw_sql(count_sql(q)).iloc[0]
                    tot += int(r["n"])
                    print(f"[{s}] {y}: n={int(r['n'])} {r['dmin']} .. {r['dmax']}", flush=True)
                print(f"[{s}] total rows: {tot}")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        gi = OUT.parent.parent / ".gitignore"  # data_reconstructed/ ignores itself (WRDS license)
        if not gi.exists():
            gi.write_text("*\n")
        man_path = OUT / "MANIFEST.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else \
            dict(files={}, environment={})
        man["environment"].update(python=sys.version.split()[0], platform=platform.platform(),
                                  wrds=getattr(wrds, "__version__", "?"),
                                  pandas=__import__("pandas").__version__,
                                  coauthor_repo_head=os.popen(f"git -C {COAUTHOR} rev-parse HEAD").read().strip(),
                                  wrapper_sha256=sha256(Path(__file__)))
        for s in sel:
            if s == "index":
                pull_index(db, jobs[s], man["files"])
            else:
                pull_to_csv(db, s, jobs[s], OUT / jobs[s]["file"], man["files"])
            man_path.write_text(json.dumps(man, indent=1))  # after each dataset
        print(f"Manifest -> {man_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
