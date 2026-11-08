"""Reference data: CNOPS medication list + pharmacy directory (Assaydaliya export).

The app loads `data/medicaments.csv` / `data/pharmacies.csv` produced by the import
scripts; if they don't exist yet it falls back to the small SAMPLE files.
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process

from .lexicon import CITIES
from .normalize import loose_latin, normalize, skeleton

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def _f(x):
    try:
        return float(str(x).replace(",", ".")) if str(x).strip() not in ("", "nan", "None") else None
    except ValueError:
        return None


def dosage_to_mg(value, unit) -> float | None:
    v = _f(value)
    if v is None:
        return None
    u = (unit or "").strip().lower()
    return {"g": v * 1000, "mg": v, "mcg": v / 1000, "µg": v / 1000, "ug": v / 1000}.get(u, v)


def pretty_dose(label: str) -> str:
    """'1 / 125 G / MG' -> '1 g / 125 mg' ; '500 MG' -> '500 mg' (CNOPS writes combos as numbers / units)."""
    if not label or re.fullmatch(r"[\sNA/]*", label.upper()):
        return ""
    m = re.match(r"^\s*([\d.,\s/]+?)\s*([A-Za-zµ%].*)?$", label)
    if not m:
        return label.lower()
    nums = [x.strip() for x in m.group(1).split("/") if x.strip()]
    units = [x.strip().lower() for x in (m.group(2) or "").split("/") if x.strip()]
    nums = [n.replace(" ", "") for n in nums]  # "1 .5" -> "1.5"
    if all(u == "na" for u in units):          # CNOPS writes NA when the unit is unknown
        units = []
    units = ["" if u == "na" else u for u in units]
    if len(units) == len(nums):
        return " / ".join(f"{n} {u}".strip() for n, u in zip(nums, units) if n).strip(" /")
    units = [u for u in units if u]
    return " / ".join(nums) + (" " + " / ".join(units) if units else "")


def label_numbers(c: dict) -> set:
    """Every number written in a variant's dosage label, also expressed in mg (for '1 g' == '1000')."""
    out = set()
    pairs = re.findall(r"(\d+(?:[.,]\d+)?)\s*([a-zµ%]*)", pretty_dose(c.get("dosage_label") or "") or "")
    for num, unit in pairs:
        v = float(num.replace(",", "."))
        out.add(v)
        mg = dosage_to_mg(v, unit)
        if mg is not None:
            out.add(mg)
    if c.get("dosage") is not None:
        out.add(c["dosage"])
    if c.get("dosage_mg") is not None:
        out.add(c["dosage_mg"])
    return out


def public_with_label(m: "Medication") -> dict:
    d = m.public()
    d["label"] = m.label()
    return d


@dataclass
class Medication:
    code: str
    name: str
    brand: str
    dci: str
    dosage: float | None
    dosage_unit: str
    form: str
    presentation: str
    ppv: float | None
    ph: float | None
    prix_br: float | None
    generic_type: str
    reimb_rate: float | None
    source: str
    reimbursable: bool | None = None      # None = unknown (e.g. SAMPLE data)
    dosage_label: str = ""
    dosage_mg: float | None = None

    def public(self) -> dict:
        return dict(self.__dict__)

    def label(self) -> str:
        """Human label distinguishing variants: 'DOLIPRANE 1 G comprimé sécable (14.00 DH)'."""
        dose = pretty_dose(self.dosage_label) or (f"{self.dosage:g} {self.dosage_unit}".strip() if self.dosage is not None else "")
        parts = [self.name]
        if dose and dose.split()[0] not in self.name:
            parts.append(dose)
        if self.form:
            parts.append(self.form)
        return " ".join(parts) + (f" ({self.ppv:.2f} DH)" if self.ppv else "")


@dataclass
class Pharmacy:
    id: str
    name: str
    city: str
    district: str
    address: str
    phone: str
    lat: float | None
    lon: float | None
    garde: bool
    hours: str
    source: str

    def public(self, distance_km: float | None = None) -> dict:
        d = dict(self.__dict__)
        if distance_km is not None:
            d["distance_km"] = round(distance_km, 2)
        return d


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _pick(primary: str, fallback: str) -> Path:
    """Real data if imported, else the SAMPLE file. USE_SAMPLE_DATA=1 forces the sample (tests)."""
    p = DATA_DIR / primary
    if os.getenv("USE_SAMPLE_DATA") == "1" or not p.exists():
        return DATA_DIR / fallback
    return p


def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class Catalog:
    meds: list[Medication] = field(default_factory=list)
    by_brand: dict[str, list[Medication]] = field(default_factory=dict)   # normalized brand -> products
    by_dci: dict[str, list[Medication]] = field(default_factory=dict)
    brand_skel: dict[str, list[str]] = field(default_factory=dict)        # skeleton -> brand keys
    dci_skel: dict[str, list[str]] = field(default_factory=dict)
    source_file: str = ""

    @classmethod
    def load(cls) -> "Catalog":
        path = _pick("medicaments.csv", "sample_medicaments.csv")
        cat = cls(source_file=path.name)
        for r in _read_csv(path):
            brand = normalize(r.get("brand") or r["name"].split()[0])
            m = Medication(
                code=r.get("code", ""), name=r["name"], brand=brand, dci=normalize(r.get("dci", "")),
                dosage=_f(r.get("dosage")), dosage_unit=(r.get("dosage_unit") or "").lower(),
                form=normalize(r.get("form", "")), presentation=r.get("presentation", ""),
                ppv=_f(r.get("ppv")), ph=_f(r.get("ph")), prix_br=_f(r.get("prix_br")),
                generic_type=r.get("generic_type", ""), reimb_rate=_f(r.get("reimb_rate")),
                source=r.get("source", "CNOPS"), dosage_label=r.get("dosage_label", ""),
                reimbursable={"1": True, "true": True, "0": False, "false": False}.get(
                    str(r.get("reimbursable", "")).strip().lower()),
            )
            m.dosage_mg = dosage_to_mg(m.dosage, m.dosage_unit)
            cat.meds.append(m)
            cat.by_brand.setdefault(brand, []).append(m)
            if m.dci:
                # index every molecule of a combination too ("SULFATE FERREUX / ACIDE FOLIQUE")
                parts = [normalize(x) for x in re.split(r"\s*[/+]\s*", r.get("dci", "")) if x.strip()]
                for key in {m.dci, *parts}:
                    cat.by_dci.setdefault(key, []).append(m)
        for src, dst in ((cat.by_brand, cat.brand_skel), (cat.by_dci, cat.dci_skel)):
            for k in src:
                sk = skeleton(k.replace(" ", ""))
                if len(sk) >= 2:
                    dst.setdefault(sk, []).append(k)
        return cat

    # ------------------------------------------------------------ lookup
    def match_span(self, span: str) -> tuple[str, str, float] | None:
        """Return (kind, key, score) for a 1-3 token span, kind in {brand, dci}."""
        is_ar = any("؀" <= c <= "ۿ" for c in span)
        compact = span.replace(" ", "")
        best = None
        if not is_ar:
            for kind, table in (("brand", self.by_brand), ("dci", self.by_dci)):
                cutoff = 84 if len(compact) >= 6 else 90      # short words need a near-exact spelling
                hit = process.extractOne(span, table.keys(), scorer=fuzz.ratio, score_cutoff=cutoff)
                if hit and (best is None or hit[1] > best[2]):
                    best = (kind, hit[0], float(hit[1]))
            if best and best[2] >= 92:
                return best
        sk = skeleton(compact)
        if len(sk) < 2:
            return best
        loose = loose_latin(compact)
        tie = lambda keys: max(keys, key=lambda k: fuzz.ratio(loose, loose_latin(k.replace(" ", ""))))
        for kind, table in (("brand", self.brand_skel), ("dci", self.dci_skel)):
            if sk in table:
                key = tie(table[sk])
                # a 3-consonant skeleton is too weak alone ("votre" ~ VITAIR): also require the vowels to agree
                agree = fuzz.ratio(loose, loose_latin(key.replace(" ", "")))
                if (len(sk) == 2 and agree < 85) or (len(sk) == 3 and agree < 80) or (not is_ar and agree < 80):
                    continue      # Latin spelling must also roughly agree on vowels ("dafalgan" is not DIFLUCAN)
                cand = (kind, key, 90.0)
            else:
                if len(sk) < 5:
                    continue
                hit = process.extractOne(sk, table.keys(), scorer=fuzz.ratio, score_cutoff=88)
                if not hit:
                    continue
                if not is_ar and max(fuzz.ratio(span, k) for k in table[hit[0]]) < 70:
                    continue   # latin word whose skeleton is merely close (e.g. "comment" ~ augmentin)
                cand = (kind, tie(table[hit[0]]), round(0.95 * hit[1], 1))
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def products(self, kind: str, key: str) -> list[Medication]:
        return (self.by_brand if kind == "brand" else self.by_dci).get(key, [])

    def search(self, q: str, limit: int = 10) -> list[dict]:
        qn = normalize(q)
        names = {i: normalize(m.name) for i, m in enumerate(self.meds)}
        hits = process.extract(qn, names, scorer=fuzz.WRatio, limit=limit, score_cutoff=60)
        return [dict(self.meds[i].public(), score=round(s, 1)) for _, s, i in hits]

    def generics_for(self, m: Medication) -> list[Medication]:
        same = [x for x in self.by_dci.get(m.dci, []) if x is not m and x.brand != m.brand]
        if m.dosage_mg is not None:
            same = [x for x in same if x.dosage_mg == m.dosage_mg]
        return sorted(same, key=lambda x: (x.ppv is None, x.ppv or 0))


@dataclass
class PharmacyDirectory:
    items: list[Pharmacy] = field(default_factory=list)
    source_file: str = ""

    @classmethod
    def load(cls) -> "PharmacyDirectory":
        path = _pick("pharmacies.csv", "sample_pharmacies.csv")
        d = cls(source_file=path.name)
        for i, r in enumerate(_read_csv(path)):
            d.items.append(Pharmacy(
                id=r.get("id") or f"PH{i:05d}", name=r.get("name", ""), city=canonical_city(r.get("city", "")),
                district=r.get("district", ""), address=r.get("address", ""), phone=r.get("phone", ""),
                lat=_f(r.get("lat")), lon=_f(r.get("lon")),
                garde=str(r.get("garde", "")).strip().lower() in ("1", "true", "oui", "yes", "y"),
                hours=r.get("hours", ""), source=r.get("source", "ASSAYDALIYA"),
            ))
        return d

    def search(self, city: str | None = None, garde: bool | None = None, lat: float | None = None,
               lon: float | None = None, district: str | None = None, limit: int = 5,
               max_km: float | None = None) -> list[dict]:
        res = self.items
        if city:
            res = [p for p in res if p.city == city]
        if garde:
            res = [p for p in res if p.garde]
        if district:
            dn = normalize(district)
            res = [p for p in res if fuzz.partial_ratio(dn, normalize(p.district + " " + p.address)) >= 85] or res
        if lat is not None and lon is not None:
            scored = [(haversine(lat, lon, p.lat, p.lon), p) for p in res if p.lat is not None]
            scored = [t for t in scored if max_km is None or t[0] <= max_km]
            scored.sort(key=lambda t: t[0])
            return [p.public(dist) for dist, p in scored[:limit]]
        return [p.public() for p in res[:limit]]

    def by_name(self, name: str, city: str | None = None) -> list[dict]:
        pool = [p for p in self.items if not city or p.city == city]
        names = {i: normalize(p.name) for i, p in enumerate(pool)}
        hits = process.extract(normalize(name), names, scorer=fuzz.token_set_ratio, limit=3, score_cutoff=80)
        return [pool[i].public() for _, _, i in hits]


# ---------------------------------------------------------------- cities
_CITY_LOOKUP = {normalize(v): c for c, vs in CITIES.items() for v in vs + [c]}


def canonical_city(raw: str) -> str:
    n = normalize(raw or "")
    if n in _CITY_LOOKUP:
        return _CITY_LOOKUP[n]
    hit = process.extractOne(n, _CITY_LOOKUP.keys(), scorer=fuzz.ratio, score_cutoff=85) if n else None
    return _CITY_LOOKUP[hit[0]] if hit else (raw or "").strip().title()
