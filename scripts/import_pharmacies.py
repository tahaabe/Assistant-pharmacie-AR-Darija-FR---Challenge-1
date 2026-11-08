"""Normalize a pharmacy export (Assaydaliya scrape, CSV / JSON / XLSX) into data/pharmacies.csv.

    python scripts/import_pharmacies.py assaydaliya_export.json
    python scripts/import_pharmacies.py pharmacies.csv --default-city Rabat

Expected fields (any of the aliases below): name, city, district, address, phone, lat, lon,
garde (on-duty flag), hours. Rows are de-duplicated on (name, city, phone).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data import canonical_city  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "pharmacies.csv"
ALIASES = {
    "name": ["name", "nom", "pharmacie", "title", "titre", "nom_pharmacie", "pharmacy"],
    "city": ["city", "ville", "localite", "town", "مدينة"],
    "district": ["district", "quartier", "zone", "secteur", "arrondissement", "neighborhood"],
    "address": ["address", "adresse", "addr", "adresse_complete", "location"],
    "phone": ["phone", "telephone", "tel", "tél", "numero", "num", "fixe", "gsm", "mobile"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "long", "longitude", "x"],
    "garde": ["garde", "de_garde", "on_duty", "is_garde", "garde_status", "nuit", "is_open"],
    "hours": ["hours", "horaires", "horaire", "opening_hours", "ouverture"],
}


def slug(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9؀-ۿ]+", "_", s).strip("_")


def load(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict):  # {"data": [...]} / {"pharmacies": [...]}
            d = next((v for v in d.values() if isinstance(v, list)), [d])
        return pd.json_normalize(d)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)
    return pd.read_csv(p, dtype=str, sep=None, engine="python", encoding="utf-8-sig")


def phone(s: str) -> str:
    digits = re.sub(r"\D", "", s or "")
    if digits.startswith("212"):
        digits = "0" + digits[3:]
    if len(digits) == 10 and digits[0] == "0":
        return f"+212 {digits[1:4]} {digits[4:7]} {digits[7:]}"
    return (s or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--default-city", default="")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    df = load(Path(a.path)).fillna("")
    by_slug = {slug(c).split("_")[-1] if "." in c else slug(c): c for c in df.columns}
    m = {f: next((by_slug[n] for n in names if n in by_slug), None) for f, names in ALIASES.items()}
    print("column mapping:", m)
    if not m["name"]:
        sys.exit(f"No name column found. Columns: {list(df.columns)}")
    get = lambda f: df[m[f]].astype(str).str.strip() if m[f] else pd.Series([""] * len(df))
    out = pd.DataFrame({
        "name": get("name"), "city": get("city").replace("", a.default_city).map(canonical_city),
        "district": get("district"), "address": get("address"), "phone": get("phone").map(phone),
        "lat": pd.to_numeric(get("lat").str.replace(",", "."), errors="coerce"),
        "lon": pd.to_numeric(get("lon").str.replace(",", "."), errors="coerce"),
        "garde": get("garde").str.lower().isin(["1", "true", "oui", "yes", "garde", "de garde", "y"]).astype(int),
        "hours": get("hours"),
    })
    out = out[out["name"] != ""].drop_duplicates(subset=["name", "city", "phone"]).reset_index(drop=True)
    out.insert(0, "id", [f"PH{i:05d}" for i in range(len(out))])
    out["source"] = "ASSAYDALIYA"
    out.to_csv(a.out, index=False, encoding="utf-8")
    print(f"wrote {len(out)} pharmacies in {out['city'].nunique()} cities "
          f"({out['lat'].notna().sum()} geolocated, {out['garde'].sum()} de garde) -> {a.out}")


if __name__ == "__main__":
    main()
