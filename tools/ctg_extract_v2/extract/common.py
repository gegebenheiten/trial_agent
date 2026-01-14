from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

WHITESPACE_RE = re.compile(r"\s+")
EUDRACT_RE = re.compile(r"\b\d{4}-\d{6}-\d{2}\b")
NCT_ID_CANDIDATES = ("nctid", "nctno", "nctnumber", "nct")
DRUG_LIKE_TYPES = {"drug", "biological", "combination product"}
PLACEBO_LIKE_KEYWORDS = ("placebo", "sham", "vehicle")


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_age_value(text: str) -> str:
    if not text:
        return ""
    lowered = text.strip().lower()
    if lowered in {"n/a", "na", "not applicable", "none"} or "no limit" in lowered:
        return ""
    match = re.search(r"\d+(?:\.\d+)?", lowered)
    if not match:
        return ""
    value = float(match.group(0))
    if "month" in lowered:
        years = value / 12
    elif "week" in lowered:
        years = value / 52
    elif "day" in lowered:
        years = value / 365.25
    else:
        years = value
    if abs(years - round(years)) < 1e-6:
        return str(int(round(years)))
    return f"{years:.2f}".rstrip("0").rstrip(".")


def normalize_gender(value: str) -> str:
    lower = normalize_whitespace(value).lower()
    if not lower:
        return ""
    if any(token in lower for token in ("n/a", "na", "not applicable", "unknown")):
        return ""
    if "male" in lower and "female" in lower:
        return "Both"
    if lower == "male":
        return "Male"
    if lower == "female":
        return "Female"
    if lower in {"all", "both"}:
        return "Both"
    return ""


def safe_int(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_whitespace(text).lower())


def arm_type_matches(arm_type: str, keywords: Iterable[str]) -> bool:
    lower = (arm_type or "").lower()
    return any(keyword in lower for keyword in keywords)


def normalize_country(name: str) -> str:
    if not name:
        return ""
    lowered = normalize_whitespace(name).lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def region_sets() -> Dict[str, set]:
    regions: Dict[str, List[str]] = {
        "NA": ["United States", "Canada"],
        "WEU": [
            "Austria",
            "Belgium",
            "France",
            "Germany",
            "Greece",
            "Italy",
            "Liechtenstein",
            "Luxembourg",
            "Monaco",
            "Netherlands",
            "Portugal",
            "Spain",
            "Switzerland",
        ],
        "EEU": [
            "Albania",
            "Belarus",
            "Bulgaria",
            "Czech Republic",
            "Czechia",
            "Estonia",
            "Hungary",
            "Latvia",
            "Lithuania",
            "Moldova",
            "Poland",
            "Romania",
            "Russia",
            "Russian Federation",
            "Serbia",
            "Slovakia",
            "Ukraine",
        ],
        "AF": [
            "Algeria",
            "Angola",
            "Benin",
            "Botswana",
            "Burkina Faso",
            "Burundi",
            "Cabo Verde",
            "Cape Verde",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Comoros",
            "Congo",
            "Congo, Republic of the",
            "Congo, The Democratic Republic of the",
            "Democratic Republic of the Congo",
            "Djibouti",
            "Egypt",
            "Equatorial Guinea",
            "Eritrea",
            "Eswatini",
            "Swaziland",
            "Ethiopia",
            "Gabon",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Ivory Coast",
            "Cote d'Ivoire",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Libya",
            "Madagascar",
            "Malawi",
            "Mali",
            "Mauritania",
            "Mauritius",
            "Morocco",
            "Mozambique",
            "Namibia",
            "Niger",
            "Nigeria",
            "Rwanda",
            "Sao Tome and Principe",
            "Senegal",
            "Seychelles",
            "Sierra Leone",
            "Somalia",
            "South Africa",
            "South Sudan",
            "Sudan",
            "Tanzania",
            "Togo",
            "Tunisia",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ],
        "AP": [
            "Afghanistan",
            "Armenia",
            "Azerbaijan",
            "Bahrain",
            "Bangladesh",
            "Bhutan",
            "Brunei",
            "Cambodia",
            "China",
            "Cyprus",
            "Georgia",
            "Hong Kong",
            "India",
            "Indonesia",
            "Iran",
            "Iraq",
            "Israel",
            "Japan",
            "Jordan",
            "Kazakhstan",
            "Kuwait",
            "Kyrgyzstan",
            "Laos",
            "Lebanon",
            "Macau",
            "Macao",
            "Malaysia",
            "Maldives",
            "Mongolia",
            "Myanmar",
            "Nepal",
            "North Korea",
            "Oman",
            "Pakistan",
            "Palestine",
            "Philippines",
            "Qatar",
            "Saudi Arabia",
            "Singapore",
            "South Korea",
            "Sri Lanka",
            "Syria",
            "Taiwan",
            "Tajikistan",
            "Thailand",
            "Timor-Leste",
            "Turkey",
            "Turkmenistan",
            "United Arab Emirates",
            "Uzbekistan",
            "Vietnam",
            "Viet Nam",
            "Yemen",
            "Australia",
            "New Zealand",
            "Papua New Guinea",
            "Fiji",
            "Solomon Islands",
            "Vanuatu",
            "Samoa",
            "Tonga",
            "Tuvalu",
            "Kiribati",
            "Nauru",
            "Palau",
            "Micronesia",
            "Micronesia, Federated States of",
            "Marshall Islands",
            "American Samoa",
            "Guam",
            "Northern Mariana Islands",
        ],
    }

    normalized: Dict[str, set] = {}
    for region, names in regions.items():
        normalized[region] = {normalize_country(n) for n in names}
    return normalized


def region_for_country(country: str, region_map: Dict[str, set]) -> str:
    key = normalize_country(country)
    if not key:
        return ""
    for region, names in region_map.items():
        if key in names:
            return region
    return ""


def parse_location_centers(root: ET.Element) -> List[tuple[str, str, str]]:
    centers: List[tuple[str, str, str]] = []
    for loc in root.findall("location"):
        name = xml_text(loc, "facility/name")
        city = xml_text(loc, "facility/address/city")
        if not city:
            city = xml_text(loc, "facility/address/state") or xml_text(
                loc, "facility/address/state_province"
            )
        country = xml_text(loc, "facility/address/country")
        if not (name or city or country):
            continue
        centers.append((name, city, country))
    seen = set()
    unique: List[tuple[str, str, str]] = []
    for center in centers:
        if center in seen:
            continue
        seen.add(center)
        unique.append(center)
    return unique


def parse_locations(root: ET.Element) -> List[str]:
    countries = []
    for loc in root.findall("location"):
        country = xml_text(loc, "facility/address/country")
        if country:
            countries.append(country)
    return countries


def parse_officials(root: ET.Element) -> List[Dict[str, str]]:
    officials = []
    for official in root.findall("overall_official"):
        role = xml_text(official, "role")
        parts = [
            xml_text(official, "first_name"),
            xml_text(official, "middle_name"),
            xml_text(official, "last_name"),
        ]
        combined = normalize_whitespace(" ".join(p for p in parts if p))
        if not combined:
            combined = xml_text(official, "last_name")
        if combined:
            officials.append({"name": combined, "role": role})
    return officials


def parse_sponsors(root: ET.Element) -> List[str]:
    sponsors: List[str] = []
    lead = xml_text(root, "sponsors/lead_sponsor/agency")
    if lead:
        sponsors.append(lead)
    for col in root.findall("sponsors/collaborator/agency"):
        name = normalize_whitespace(col.text or "")
        if name:
            sponsors.append(name)
    return sponsors


def normalize_intervention_model(value: str) -> str:
    lower = normalize_whitespace(value).lower()
    if not lower:
        return ""
    if any(token in lower for token in ("n/a", "na", "not applicable", "unknown", "not provided")):
        return ""
    return re.sub(r"[\s\-_]+", "", lower)


def infer_blinding(masking: str) -> tuple[str, str]:
    if not masking:
        return "", ""
    lower = masking.lower()
    if any(token in lower for token in ("n/a", "na", "not applicable", "unknown")):
        return "", ""
    if "none" in lower or "open" in lower:
        return "No", "0"
    if "single" in lower:
        return "Yes", "1"
    if "double" in lower or "triple" in lower or "quadruple" in lower:
        if "triple" in lower:
            return "Yes", "3"
        if "quad" in lower:
            return "Yes", "4"
        return "Yes", "2"
    return "Yes", ""


def extract_eudract_id(text: str) -> str:
    if not text:
        return ""
    match = EUDRACT_RE.search(text)
    return match.group(0) if match else ""


def parse_secondary_ids(root: ET.Element) -> List[str]:
    return xml_texts(root, "id_info/secondary_id")


def is_placebo_like_name(name: str) -> bool:
    lower = (name or "").lower()
    return any(keyword in lower for keyword in PLACEBO_LIKE_KEYWORDS)


def join_values(values: Iterable[str], delimiter: str = "; ") -> str:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = normalize_whitespace(str(value))
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return delimiter.join(ordered)


def xml_text(root: ET.Element, path: str) -> str:
    return normalize_whitespace(root.findtext(path) or "")


def xml_texts(root: ET.Element, path: str) -> List[str]:
    values: List[str] = []
    for el in root.findall(path):
        value = normalize_whitespace(el.text or "")
        if value:
            values.append(value)
    return values


def normalize_intervention_type(value: str) -> str:
    lower = normalize_whitespace(value).lower()
    if not lower:
        return ""
    lowered = lower.replace("_", " ").replace("-", " ").replace("/", " ")
    lowered = WHITESPACE_RE.sub(" ", lowered).strip()
    if "combination" in lowered and "product" in lowered:
        return "combination product"
    if "biologic" in lowered or "biological" in lowered or "vaccine" in lowered:
        return "biological"
    if "drug" in lowered:
        return "drug"
    return lowered


def is_drug_like(iv_type: str) -> bool:
    return normalize_intervention_type(iv_type) in DRUG_LIKE_TYPES


def parse_interventions(root: ET.Element) -> List[Dict[str, List[str] | str]]:
    interventions = []
    for iv in root.findall("intervention"):
        name = xml_text(iv, "intervention_name")
        other_names = [
            normalize_whitespace(o.text or "")
            for o in iv.findall("other_name")
            if normalize_whitespace(o.text or "")
        ]
        labels = [
            normalize_whitespace(l.text or "")
            for l in iv.findall("arm_group_label")
            if normalize_whitespace(l.text or "")
        ]
        interventions.append(
            {
                "type": normalize_intervention_type(xml_text(iv, "intervention_type")),
                "name": name,
                "description": xml_text(iv, "description"),
                "other_names": other_names,
                "labels": labels,
            }
        )
    return interventions


def parse_arm_groups(root: ET.Element) -> List[Dict[str, str]]:
    arms = []
    for arm in root.findall("arm_group"):
        arms.append(
            {
                "label": xml_text(arm, "arm_group_label"),
                "type": xml_text(arm, "arm_group_type"),
                "description": xml_text(arm, "description"),
            }
        )
    return arms


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def parse_nct_ids(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s]+", raw.strip())
    seen = set()
    ordered: List[str] = []
    for part in parts:
        if not part:
            continue
        value = part.upper()
        if not value.startswith("NCT"):
            value = f"NCT{value}"
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def detect_nct_id_column(fieldnames: Iterable[str], preferred: str = "") -> str:
    if preferred:
        preferred_key = normalize_header(preferred)
        for name in fieldnames:
            if normalize_header(name) == preferred_key:
                return name
        raise ValueError(f"NCT ID column '{preferred}' not found in CSV header.")

    normalized = {normalize_header(name): name for name in fieldnames if name}
    for candidate in NCT_ID_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("NCT ID column not found in CSV header; pass --nct-id-col.")


def load_nct_ids_from_csv(csv_path: Path, column: str = "", limit: int = 0) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing NCT CSV: {csv_path}")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"NCT CSV has no header: {csv_path}")
        col_name = detect_nct_id_column(reader.fieldnames, column)
        seen = set()
        ordered: List[str] = []
        for row in reader:
            raw = row.get(col_name, "")
            for nct_id in parse_nct_ids(raw or ""):
                if nct_id in seen:
                    continue
                seen.add(nct_id)
                ordered.append(nct_id)
                if limit and len(ordered) >= limit:
                    return ordered
    return ordered


def merge_nct_ids(primary: List[str], secondary: List[str], limit: int = 0) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for source in (primary, secondary):
        for nct_id in source:
            if nct_id in seen:
                continue
            seen.add(nct_id)
            ordered.append(nct_id)
            if limit and len(ordered) >= limit:
                return ordered
    return ordered


def xml_path_for_id(xml_root: Path, nct_id: str) -> Path:
    prefix = nct_id[:7] + "xxxx"
    return xml_root / prefix / f"{nct_id}.xml"


def iter_xml_files(xml_root: Path, nct_ids: List[str]) -> Iterable[Path]:
    if nct_ids:
        for nct_id in nct_ids:
            path = xml_path_for_id(xml_root, nct_id)
            if path.exists():
                yield path
            else:
                print(f"[WARN] XML not found for {nct_id}: {path}")
        return
    for path in xml_root.rglob("NCT*.xml"):
        yield path


def suffix_for_ncts(nct_ids: List[str]) -> str:
    if not nct_ids:
        return ""
    if len(nct_ids) == 1:
        return nct_ids[0]
    if len(nct_ids) <= 3:
        return "_".join(nct_ids)
    return f"{nct_ids[0]}_plus{len(nct_ids) - 1}"


def set_if_present(row: Dict[str, str], field: str, value: str) -> None:
    if field in row and value:
        row[field] = value


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_whitespace(text).lower()).strip("_")
