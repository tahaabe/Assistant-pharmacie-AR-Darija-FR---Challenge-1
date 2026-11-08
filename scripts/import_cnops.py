"""Convert the CNOPS / ANAM medication list (xlsx, xls or csv) into data/medicaments.csv.

    python scripts/import_cnops.py path/to/liste_medicaments.xlsx
    python scripts/import_cnops.py file.xlsx --sheet 0 --reimbursable-list   # file IS the reimbursable list

Column names vary between releases, so they are matched through aliases. The script prints
the mapping it used and stops with the list of columns if the drug name can't be found:
just add the missing header to ALIASES.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "medicaments.csv"

ALIASES = {
    "code": ["code", "code_ean", "ean", "code_barre", "codebarre", "code_med", "code_medicament", "id"],
    "name": ["nom", "nom_commercial", "specialite", "designation", "medicament", "libelle", "nom_du_medicament", "name"],
    "dci": ["dci", "dci1", "substance_active", "principe_actif", "molecule", "composition"],
    "dosage": ["dosage", "dosage1", "dose"],
    "dosage_unit": ["unite_dosage", "unite_dosage1", "unite", "udosage"],
    "form": ["forme", "forme_pharmaceutique", "forme_galenique", "form"],
    "presentation": ["presentation", "conditionnement", "boite"],
    "ppv": ["ppv", "prix_public", "prix_public_de_vente", "prix_ppv", "prix"],
    "ph": ["ph", "prix_hopital", "prix_hospitalier"],
    "prix_br": ["prix_br", "base_remboursement", "prix_base_remboursement", "pbr", "br", "base_de_remboursement"],
    "generic_type": ["princeps_generique", "princeps/generique", "type", "p_g", "statut", "generique"],
    "reimb_rate": ["taux_remboursement", "taux_de_remboursement", "taux", "taux_rembt"],
    "reimbursable": ["remboursable", "rembourse", "remboursement", "amo"],
}


def slug(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9/]+", "_", s).strip("_")


def map_columns(cols) -> dict[str, str]:
    by_slug = {slug(c): c for c in cols}
    mapping = {}
    for field, names in ALIASES.items():
        for n in names:
            if n in by_slug:
                mapping[field] = by_slug[n]
                break
    return mapping


def find_header_row(raw: pd.DataFrame) -> int:
    """Some official sheets have a title block above the real header."""
    for i in range(min(20, len(raw))):
        row = [slug(x) for x in raw.iloc[i].tolist()]
        if any(x in row for x in ALIASES["name"]):
            return i
    return 0


def split_dosage(name: str) -> tuple[str, str]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(mg|g|ml|µg|mcg|ui|%)\b", name, re.I)
    return (m.group(1).replace(",", "."), m.group(2).lower()) if m else ("", "")


def brand_of(name: str) -> str:
    """Brand = first word of the product name ("DOLIPRANE VITAMINE C" -> DOLIPRANE), so that a patient
    saying "doliprane" reaches every DOLIPRANE product; very short first words keep the full head."""
    head = re.split(r"\s\d|\s(?:comprim|gelul|sirop|inject|sachet|suppo|pommade|creme|collyre|solution|poudre)",
                    name, maxsplit=1, flags=re.I)[0].strip().upper()
    first = re.split(r"[\s/\-]+", head)[0]
    if first in GENERIC_WORDS:          # "ACIDE FOLIQUE ...", "VITAMINE C ..." are not brands
        return head
    return first if len(first) >= 4 and first.isalpha() else head


GENERIC_WORDS: set[str] = set()


def first_number(v: str) -> str:
    m = re.search(r"\d+(?:[.,]\d+)?", v or "")
    return m.group(0).replace(",", ".") if m else ""


def first_unit(u: str) -> str:
    return (u or "").split(" / ")[0].strip().lower().replace("µg", "mcg")


def truthy(v) -> str:
    s = slug(v)
    if s in ("oui", "o", "yes", "y", "1", "true", "r", "remboursable"):
        return "1"
    if s in ("non", "n", "no", "0", "false", "nr", "non_remboursable"):
        return "0"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--reimbursable-list", action="store_true",
                    help="every row in this file is reimbursable (e.g. the CNOPS 'liste des médicaments remboursables')")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    p = Path(a.path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet
        raw = pd.read_excel(p, sheet_name=sheet, header=None, dtype=str)
        h = find_header_row(raw)
        df = raw.iloc[h + 1:].copy()
        df.columns = [str(c) for c in raw.iloc[h].tolist()]
    else:
        df = pd.read_csv(p, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    df = df.dropna(how="all")
    m = map_columns(df.columns)
    print("column mapping:", m)
    if "name" not in m:
        sys.exit(f"Could not find the drug-name column. Columns are: {list(df.columns)}")

    out = pd.DataFrame()
    for f in ["code", "name", "dci", "dosage", "dosage_unit", "form", "presentation", "ppv", "ph", "prix_br",
              "generic_type", "reimb_rate", "reimbursable"]:
        out[f] = df[m[f]].fillna("").astype(str).str.strip() if f in m else ""
    out = out[out["name"] != ""]
    # fill dosage from the name when there is no dedicated column
    miss = out["dosage"] == ""
    parsed = out.loc[miss, "name"].map(split_dosage)
    out.loc[miss, "dosage"] = parsed.map(lambda t: t[0])
    out.loc[miss & (out["dosage_unit"] == ""), "dosage_unit"] = parsed.map(lambda t: t[1])
    # a first word that is also a molecule / chemistry word (ACIDE, SODIUM, VITAMINE…) is not a brand
    GENERIC_WORDS.update(w for d in out["dci"] for w in re.split(r"[^A-Z]+", d.upper()) if len(w) >= 3)
    GENERIC_WORDS.update({"VITAMINE", "SERUM", "EAU", "SOLUTION", "SIROP", "CREME", "POMMADE", "GEL", "LAIT"})
    out["brand"] = out["name"].map(brand_of)
    # combined products ("500 / 150", "MG / MG"): keep the full label, index on the first molecule's dose
    out["dosage_label"] = (out["dosage"] + " " + out["dosage_unit"]).str.strip()
    out["dosage"] = out["dosage"].map(first_number)
    out["dosage_unit"] = out["dosage_unit"].map(first_unit)
    for c in ("ppv", "ph", "prix_br", "reimb_rate"):
        out[c] = out[c].str.replace(",", ".", regex=False).str.replace(r"[^\d.]", "", regex=True)
    if a.reimbursable_list:
        out["reimbursable"] = "1"
    else:
        out["reimbursable"] = out["reimbursable"].map(truthy)
        # CNOPS referential: TAUX_REMBOURSEMENT 0% = listed but not reimbursed, >0% = reimbursed
        rate = pd.to_numeric(out["reimb_rate"], errors="coerce")
        out.loc[(out["reimbursable"] == "") & (rate > 0), "reimbursable"] = "1"
        out.loc[(out["reimbursable"] == "") & (rate == 0), "reimbursable"] = "0"
    out["generic_type"] = out["generic_type"].str.upper().str[:1]
    out.loc[out["code"] == "", "code"] = [f"C{i:06d}" for i in range((out["code"] == "").sum())]
    out["source"] = "CNOPS"
    out = out.drop_duplicates(subset=["code"])
    cols = ["code", "name", "brand", "dci", "dosage", "dosage_unit", "dosage_label", "form", "presentation", "ppv", "ph",
            "prix_br", "generic_type", "reimb_rate", "reimbursable", "source"]
    out[cols].to_csv(a.out, index=False, encoding="utf-8")
    print(f"wrote {len(out)} medications ({out['brand'].nunique()} brands) -> {a.out}")
    print(out[cols].head(5).to_string())


if __name__ == "__main__":
    main()
