"""Download the GWTC parameter-estimation releases and list their EOB runs.

Releases (Zenodo), oldest first; an event in several releases is taken from the latest:
    GWTC-2.1  O1-O3a  SEOBNRv4PHM (RIFT / LALInference)
    GWTC-3    O3b     SEOBNRv4PHM (RIFT / LALInference)
    GWTC-4.1  O4a     SEOBNRv5PHM (bilby, gwsignal: the GW250114 code path)
    GWTC-5.0  O4b     SEOBNRv5PHM (RIFT)
GWTC-4.1 supersedes GWTC-4.0. GWTC-1 events are in GWTC-2.1.

The full files are ~125 GB together; `fetch` downloads one file at a time, verifies its
md5, keeps only the EOB runs in data/slim/<event>.h5 and deletes the original
(--keep-full to keep it).

Usage:
    python catalog.py manifest            # list the files on Zenodo -> data/manifest.json
    python catalog.py fetch [--release GWTC-4.1] [--events GW231231_154016 ...] [--dry-run]
    python catalog.py inventory           # EOB runs of the slim files -> data/inventory.csv
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import h5py
import numpy as np

DATA = Path(__file__).parent / "data"
SLIM = DATA / "slim"
MANIFEST = DATA / "manifest.json"
INVENTORY = DATA / "inventory.csv"

# name: (Zenodo record ids, regex selecting one file per event)
RELEASES = {
    "GWTC-2.1": ([6513631], r"_PEDataRelease_mixed_nocosmo\.h5$"),
    "GWTC-3": ([8177023], r"_PEDataRelease_mixed_nocosmo\.h5$"),
    "GWTC-4.1": ([20275769], r"-combined_PEDataRelease\.hdf5$"),
    "GWTC-5.0": ([20348005, 20348006], r"-combined_PEDataRelease\.hdf5$"),
}
EVENT_NAME = re.compile(r"(GW\d{6}_\d{6})")
EOB_APPROXIMANTS = ("SEOBNRv5PHM", "SEOBNRv4PHM")


def zenodo_files(record):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as response:
        return json.load(response)["files"]


def build_manifest():
    """One entry per event, from the latest release containing it."""
    events = {}
    for release, (records, pattern) in RELEASES.items():
        for record in records:
            for f in zenodo_files(record):
                name = EVENT_NAME.search(f["key"])
                if name and re.search(pattern, f["key"]):
                    events[name.group(1)] = dict(
                        event=name.group(1), release=release, record=record, file=f["key"],
                        size=f["size"], md5=f["checksum"].removeprefix("md5:"),
                        url=f["links"]["self"])
    manifest = sorted(events.values(), key=lambda e: e["event"])
    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    by_release = {r: sum(e["release"] == r for e in manifest) for r in RELEASES}
    print(f"{len(manifest)} events, {sum(e['size'] for e in manifest) / 1e9:.0f} GB: {by_release}")
    return manifest


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else build_manifest()


def download(entry, path):
    """Download with resume, then check the md5."""
    done = path.stat().st_size if path.exists() else 0
    if done < entry["size"]:
        request = urllib.request.Request(entry["url"], headers={"Range": f"bytes={done}-"})
        with urllib.request.urlopen(request) as response, open(path, "ab") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            md5.update(chunk)
    if md5.hexdigest() != entry["md5"]:
        path.unlink()
        raise IOError(f"{path.name}: md5 mismatch (deleted, retry)")


def eob_labels(f):
    return [label for label in f if any(a in label for a in EOB_APPROXIMANTS)]


def slim(full, out):
    """Copy the EOB runs (all their groups) and the file metadata to `out`."""
    with h5py.File(full, "r") as f, h5py.File(out, "w") as g:
        for label in eob_labels(f) + [k for k in ("history", "version") if k in f]:
            f.copy(label, g)


def fetch(entries, keep_full=False, dry_run=False):
    SLIM.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries if not (SLIM / f"{e['event']}.h5").exists()]
    print(f"{len(todo)} files to fetch, {sum(e['size'] for e in todo) / 1e9:.1f} GB")
    if dry_run:
        for e in todo:
            print(f"  {e['event']}  {e['release']}  {e['size'] / 1e6:.0f} MB")
        return
    for e in todo:
        full = DATA / e["file"]
        print(f"{e['event']} ({e['size'] / 1e6:.0f} MB)", flush=True)
        try:
            download(e, full)
        except Exception as err:  # keep going, report
            print(f"  failed: {err!r}")
            continue
        slim(full, SLIM / f"{e['event']}.h5")
        if not keep_full:
            full.unlink()


# --- run settings -------------------------------------------------------------------------

def _config_value(section, key):
    value = section[key][()]
    value = value[0] if isinstance(value, np.ndarray) else value
    return value.decode() if isinstance(value, bytes) else str(value)


def _ifo_dict(text):
    """'{ H1:20,L1:20, waveform: 10. }' -> {'H1': 20.0, 'L1': 20.0, 'waveform': 10.0}"""
    text = text.strip()
    if not text.startswith("{"):
        return {"all": float(text)}
    items = [kv.split(":") for kv in text.strip("{} ").split(",") if kv.strip()]
    return {k.strip(): float(v) for k, v in items}


def run_info(run):
    """Pipeline and waveform settings of one PESummary run group."""
    config = run["config_file"]
    approximant = next((a for a in EOB_APPROXIMANTS if a in run.name), "?")
    info = dict(approximant=approximant, n_samples=len(run["posterior_samples"]))
    if "config" in config and "sampler" in config["config"]:
        c = config["config"]
        minimum = _ifo_dict(_config_value(c, "minimum-frequency"))
        arguments = ast.literal_eval(_config_value(c, "waveform-arguments-dict") or "{}") or {}
        info.update(
            pipeline="bilby",
            source_model=_config_value(c, "frequency-domain-source-model"),
            settings=dict(
                reference_frequency=float(_config_value(c, "reference-frequency")),
                minimum_frequency=minimum.get("waveform", min(minimum.values())),
                # bilby_pipe passes the largest of the detectors' maximum frequencies
                maximum_frequency=max(_ifo_dict(_config_value(c, "maximum-frequency")).values()),
                duration=float(_config_value(c, "duration")),
                lmax_nyquist=arguments.get("lmax_nyquist", 2),
            ))
    elif "rift-pseudo-pipe" in config:
        e = config["engine"]
        info.update(pipeline="RIFT", settings=dict(
            reference_frequency=float(_config_value(e, "fref")),
            minimum_frequency=float(_config_value(e, "fmin-template")),
            l_max=int(_config_value(config["rift-pseudo-pipe"], "l-max")),
        ))
    elif "engine" in config:
        # LALInference-style ini: LALInference itself or (in O3) RIFT
        e = config["engine"]
        info.update(pipeline="lalinference-ini", settings=dict(
            reference_frequency=float(_config_value(e, "fref")),
            minimum_frequency=float(_config_value(e, "fmin-template")),
            sampling_rate=float(_config_value(e, "srate")),
        ))
    else:
        info.update(pipeline="unknown", settings={})
    return info


def supported(info):
    """Runs whose waveform call is reproduced exactly (see `relabel_catalog.py`)."""
    return (info["approximant"] == "SEOBNRv5PHM" and info["pipeline"] == "bilby"
            and info.get("source_model") == "gwsignal_binary_black_hole")


def inventory():
    manifest = {e["event"]: e for e in load_manifest()}
    rows = []
    for path in sorted(SLIM.glob("GW*.h5")):
        with h5py.File(path, "r") as f:
            for label in eob_labels(f):
                info = run_info(f[label])
                samples = f[label]["posterior_samples"]
                rows.append(dict(
                    event=path.stem, release=manifest.get(path.stem, {}).get("release", "?"),
                    label=label, approximant=info["approximant"], pipeline=info["pipeline"],
                    supported=supported(info), n_samples=info["n_samples"],
                    total_mass=float(np.median(samples["total_mass"])),
                    mass_ratio=float(np.median(samples["mass_ratio"])),
                    settings=json.dumps(info["settings"])))
    with open(INVENTORY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(r["supported"] for r in rows)
    print(f"{len(rows)} EOB runs in {len(set(r['event'] for r in rows))} events; {n_ok} supported")
    for key in ("pipeline", "approximant"):
        values = [r[key] for r in rows]
        print(f"  by {key}:", {v: values.count(v) for v in sorted(set(values))})


def load_inventory():
    with open(INVENTORY) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["supported"] = r["supported"] == "True"
        r["settings"] = json.loads(r["settings"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    p = sub.add_parser("fetch")
    p.add_argument("--release", choices=list(RELEASES), nargs="*")
    p.add_argument("--events", nargs="*")
    p.add_argument("--keep-full", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("inventory")
    args = parser.parse_args()

    if args.command == "manifest":
        build_manifest()
    elif args.command == "fetch":
        entries = load_manifest()
        if args.release:
            entries = [e for e in entries if e["release"] in args.release]
        if args.events:
            entries = [e for e in entries if e["event"] in args.events]
        fetch(entries, args.keep_full, args.dry_run)
    elif args.command == "inventory":
        inventory()


if __name__ == "__main__":
    main()
