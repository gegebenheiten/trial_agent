from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET
import json
import re

from .common import (
    is_drug_like,
    is_placebo_like_name,
    join_values,
    normalize_label,
    normalize_whitespace,
    parse_arm_groups,
    parse_interventions,
    set_if_present,
    xml_text,
)

def outcome_type_label(text: str) -> str:
    lower = (text or "").strip().lower()
    if not lower:
        return ""
    if "key" in lower and "secondary" in lower:
        return "Key Secondary"
    if "primary" in lower:
        return "Primary"
    if "secondary" in lower:
        return "Secondary"
    return ""


def parse_group_list(section: ET.Element | None) -> List[Dict[str, str]]:
    if section is None:
        return []
    groups: List[Dict[str, str]] = []
    for group in section.findall("group_list/group"):
        groups.append(
            {
                "group_id": (group.attrib.get("group_id") or "").strip(),
                "title": xml_text(group, "title"),
                "description": xml_text(group, "description"),
            }
        )
    return groups


GROUP_ARM_MAPPING_FIELDS = [
    "StudyID",
    "Arm_ID",
    "Arm_Title",
    "Flow_Group_ID",
    "Flow_Group_Title",
    "Baseline_Group_ID",
    "Baseline_Group_Title",
    "Outcome_Group_ID",
    "Outcome_Group_Title",
    "EP_Name",
    "EP_type",
]


@dataclass
class CanonicalGroup:
    arm_id: str = ""
    arm_index: int = 0
    arm_type: str = ""
    arm_title: str = ""
    arm_desc: str = ""
    flow_group_id: str = ""
    flow_title: str = ""
    flow_desc: str = ""
    baseline_group_id: str = ""
    baseline_title: str = ""
    baseline_desc: str = ""

    def best_title(self) -> str:
        return self.arm_title or self.flow_title or self.baseline_title

    def best_desc(self) -> str:
        return self.flow_desc or self.baseline_desc or self.arm_desc


@dataclass
class GroupAlignment:
    groups: List[CanonicalGroup] = field(default_factory=list)
    by_title: Dict[str, CanonicalGroup] = field(default_factory=dict)
    by_desc: Dict[str, CanonicalGroup] = field(default_factory=dict)
    by_flow_id: Dict[str, CanonicalGroup] = field(default_factory=dict)
    by_baseline_id: Dict[str, CanonicalGroup] = field(default_factory=dict)
    by_numeric_id: Dict[str, List[CanonicalGroup]] = field(default_factory=dict)
    next_fallback_index: int = 1

    def _register_key(self, key: str, group: CanonicalGroup, mapping: Dict[str, CanonicalGroup]) -> None:
        if key and key not in mapping:
            mapping[key] = group

    def register(self, group: CanonicalGroup, title: str = "", desc: str = "", flow_id: str = "", baseline_id: str = "") -> None:
        self._register_key(normalize_label(title), group, self.by_title)
        self._register_key(normalize_label(desc), group, self.by_desc)
        if flow_id:
            self._register_key(flow_id, group, self.by_flow_id)
            self._register_numeric(flow_id, group)
        if baseline_id:
            self._register_key(baseline_id, group, self.by_baseline_id)
            self._register_numeric(baseline_id, group)

    def _register_numeric(self, group_id: str, group: CanonicalGroup) -> None:
        key = extract_group_number(group_id)
        if not key:
            return
        existing = self.by_numeric_id.setdefault(key, [])
        if group not in existing:
            existing.append(group)

    def match(self, group_id: str, title: str, desc: str) -> Optional[CanonicalGroup]:
        title_key = normalize_label(title)
        if title_key and title_key in self.by_title:
            return self.by_title[title_key]
        if group_id:
            if group_id in self.by_flow_id:
                return self.by_flow_id[group_id]
            if group_id in self.by_baseline_id:
                return self.by_baseline_id[group_id]
            numeric_key = extract_group_number(group_id)
            if numeric_key:
                candidates = self.by_numeric_id.get(numeric_key, [])
                if len(candidates) == 1:
                    return candidates[0]
        desc_key = normalize_label(desc)
        if desc_key and desc_key in self.by_desc:
            return self.by_desc[desc_key]
        return None

    def new_group(self) -> CanonicalGroup:
        group = CanonicalGroup()
        self.groups.append(group)
        return group

    def ensure_arm_id(self, group: CanonicalGroup, nct_id: str) -> None:
        if group.arm_id:
            return
        if group.flow_group_id:
            group.arm_id = f"{nct_id}_{group.flow_group_id}"
            return
        if group.baseline_group_id:
            group.arm_id = f"{nct_id}_{group.baseline_group_id}"
            return
        if group.arm_index:
            group.arm_id = f"{nct_id}_A{group.arm_index}"
            return
        group.arm_id = f"{nct_id}_A{self.next_fallback_index}"
        self.next_fallback_index += 1

    def assign_arm_ids(self, nct_id: str) -> None:
        for group in self.groups:
            self.ensure_arm_id(group, nct_id)

    def add_flow_group(self, group_id: str, title: str, desc: str) -> None:
        group = self.match(group_id, title, desc)
        if group is None:
            group = self.new_group()
        if group_id and not group.flow_group_id:
            group.flow_group_id = group_id
        if title and not group.flow_title:
            group.flow_title = title
        if desc and not group.flow_desc:
            group.flow_desc = desc
        self.register(group, title=title, desc=desc, flow_id=group_id)

    def add_baseline_group(self, group_id: str, title: str, desc: str) -> None:
        group = self.match(group_id, title, desc)
        if group is None:
            group = self.new_group()
        if group_id and not group.baseline_group_id:
            group.baseline_group_id = group_id
        if title and not group.baseline_title:
            group.baseline_title = title
        if desc and not group.baseline_desc:
            group.baseline_desc = desc
        self.register(group, title=title, desc=desc, baseline_id=group_id)

    def add_arm_group(self, label: str, desc: str, arm_type: str, index: int) -> None:
        group = self.match("", label, desc)
        if group is None:
            group = self.new_group()
        if index and not group.arm_index:
            group.arm_index = index
            self._register_numeric(str(index), group)
        if arm_type and not group.arm_type:
            group.arm_type = arm_type
        if label and not group.arm_title:
            group.arm_title = label
        if desc and not group.arm_desc:
            group.arm_desc = desc
        self.register(group, title=label, desc=desc)

    def add_outcome_group(self, group_id: str, title: str, desc: str) -> CanonicalGroup:
        group = self.match(group_id, title, desc)
        if group is None:
            group = self.new_group()
        if title and not group.arm_title:
            group.arm_title = title
        if desc and not group.arm_desc:
            group.arm_desc = desc
        self.register(group, title=title, desc=desc)
        return group


def build_group_alignment(root: ET.Element, nct_id: str) -> GroupAlignment:
    alignment = GroupAlignment()
    results = root.find("clinical_results")
    if results is None:
        return alignment

    flow_groups = parse_group_list(results.find("participant_flow"))
    for group in flow_groups:
        alignment.add_flow_group(
            group.get("group_id") or "",
            group.get("title") or "",
            group.get("description") or "",
        )

    baseline_groups = parse_group_list(results.find("baseline"))
    for group in baseline_groups:
        alignment.add_baseline_group(
            group.get("group_id") or "",
            group.get("title") or "",
            group.get("description") or "",
        )

    for idx, arm in enumerate(parse_arm_groups(root), start=1):
        alignment.add_arm_group(
            str(arm.get("label") or "").strip(),
            str(arm.get("description") or "").strip(),
            normalize_whitespace(str(arm.get("type") or "")),
            idx,
        )

    alignment.assign_arm_ids(nct_id)
    return alignment


def group_pop_text(title: str, description: str) -> str:
    title = normalize_whitespace(title)
    description = normalize_whitespace(description)
    if title and description:
        return f"{title}: {description}"
    return title or description


def extract_group_number(group_id: str) -> str:
    match = re.search(r"\d+", group_id or "")
    return match.group(0) if match else ""


def collect_measure_entries(measure: ET.Element) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen: set = set()

    def add_entry(category_title: str, measurement: ET.Element) -> None:
        if id(measurement) in seen:
            return
        seen.add(id(measurement))
        group_id = (measurement.attrib.get("group_id") or "").strip()
        value = (measurement.attrib.get("value") or "").strip()
        if not value:
            value = normalize_whitespace(measurement.text or "")
        spread = (measurement.attrib.get("spread") or "").strip()
        lower = (measurement.attrib.get("lower_limit") or "").strip()
        upper = (measurement.attrib.get("upper_limit") or "").strip()
        entries.append(
            {
                "category": normalize_whitespace(category_title or ""),
                "group_id": group_id,
                "value": value,
                "spread": spread,
                "lower": lower,
                "upper": upper,
            }
        )

    for cls in measure.findall("class_list/class"):
        cls_title = xml_text(cls, "title")
        categories = cls.findall("category_list/category")
        if categories:
            for cat in categories:
                cat_title = xml_text(cat, "title") or cls_title
                for measurement in cat.findall("measurement_list/measurement"):
                    add_entry(cat_title, measurement)
        else:
            for measurement in cls.findall("measurement_list/measurement"):
                add_entry(cls_title, measurement)

    for cat in measure.findall("category_list/category"):
        cat_title = xml_text(cat, "title")
        for measurement in cat.findall("measurement_list/measurement"):
            add_entry(cat_title, measurement)

    for measurement in measure.findall("measurement_list/measurement"):
        add_entry("", measurement)

    return entries


def parse_analyzed_nodes(nodes: List[ET.Element]) -> List[Dict[str, object]]:
    analyzed_items: List[Dict[str, object]] = []
    for analyzed in nodes:
        counts: List[Dict[str, str]] = []
        for count in analyzed.findall("count_list/count"):
            group_id = (count.attrib.get("group_id") or "").strip()
            value = (count.attrib.get("value") or "").strip()
            if not value:
                value = normalize_whitespace(count.text or "")
            if not value and not group_id:
                continue
            entry: Dict[str, str] = {}
            if group_id:
                entry["group_id"] = group_id
            if value:
                entry["value"] = value
            counts.append(entry)
        if not counts:
            continue
        item: Dict[str, object] = {"counts": counts}
        units = xml_text(analyzed, "units")
        scope = xml_text(analyzed, "scope")
        if units:
            item["units"] = units
        if scope:
            item["scope"] = scope
        analyzed_items.append(item)
    return analyzed_items


def parse_outcome_measures(outcome: ET.Element) -> List[Dict[str, object]]:
    measures: List[Dict[str, object]] = []
    measure_nodes = outcome.findall("measure_list/measure")
    if not measure_nodes:
        single_measure = outcome.find("measure")
        if single_measure is not None:
            measure_nodes = [single_measure]

    outcome_analyzed = parse_analyzed_nodes(outcome.findall("analyzed_list/analyzed"))

    for measure in measure_nodes:
        entry: Dict[str, object] = {}
        title = xml_text(measure, "title")
        units = xml_text(measure, "units")
        param = xml_text(measure, "param")
        dispersion = xml_text(measure, "dispersion")
        if title:
            entry["measure_title"] = title
        if units:
            entry["units"] = units
        if param:
            entry["param"] = param
        if dispersion:
            entry["dispersion"] = dispersion

        values = [
            item
            for item in collect_measure_entries(measure)
            if item.get("group_id")
            or item.get("value")
            or item.get("spread")
            or item.get("lower")
            or item.get("upper")
        ]
        if values:
            entry["values"] = values

        analyzed_items = parse_analyzed_nodes(measure.findall("analyzed_list/analyzed"))
        if not analyzed_items and outcome_analyzed:
            analyzed_items = outcome_analyzed
        if analyzed_items:
            entry["analyzed"] = analyzed_items

        if entry:
            measures.append(entry)
    return measures


def collect_measure_group_ids(measures: List[Dict[str, object]]) -> set:
    group_ids: set = set()
    for measure in measures:
        for value in measure.get("values") or []:
            if isinstance(value, dict) and value.get("group_id"):
                group_ids.add(value["group_id"])
        for analyzed in measure.get("analyzed") or []:
            for count in analyzed.get("counts") or []:
                if isinstance(count, dict) and count.get("group_id"):
                    group_ids.add(count["group_id"])
    return group_ids


def build_group_arm_mapping_rows(root: ET.Element, nct_id: str) -> List[Dict[str, str]]:
    results = root.find("clinical_results")
    if results is None:
        return []

    alignment = build_group_alignment(root, nct_id)
    rows: List[Dict[str, str]] = []
    used_groups: set = set()

    outcomes = results.findall("outcome_list/outcome")
    for outcome in outcomes:
        outcome_title = xml_text(outcome, "title")
        outcome_type = outcome_type_label(xml_text(outcome, "type"))
        measures = parse_outcome_measures(outcome)
        groups = parse_group_list(outcome)

        group_map = {group.get("group_id") or f"title:{normalize_label(group.get('title', ''))}": group for group in groups}
        for group_id in sorted(collect_measure_group_ids(measures)):
            if group_id not in group_map:
                group_map[group_id] = {"group_id": group_id, "title": "", "description": ""}
        if not group_map:
            group_map = {"overall": {"group_id": "", "title": "Overall", "description": ""}}

        outcome_cache: Dict[str, CanonicalGroup] = {}
        for group in group_map.values():
            group_id = (group.get("group_id") or "").strip()
            group_title = str(group.get("title") or "").strip()
            group_desc = str(group.get("description") or "").strip()

            canonical = None
            if group_id and group_id in outcome_cache:
                canonical = outcome_cache[group_id]
            else:
                canonical = alignment.match(group_id, group_title, group_desc)
                if canonical is None:
                    canonical = alignment.add_outcome_group(group_id, group_title, group_desc)
                alignment.ensure_arm_id(canonical, nct_id)
                if group_id:
                    outcome_cache[group_id] = canonical

            row = {field: "" for field in GROUP_ARM_MAPPING_FIELDS}
            row["StudyID"] = nct_id
            row["Arm_ID"] = canonical.arm_id
            row["Arm_Title"] = canonical.best_title()
            row["Flow_Group_ID"] = canonical.flow_group_id
            row["Flow_Group_Title"] = canonical.flow_title
            row["Baseline_Group_ID"] = canonical.baseline_group_id
            row["Baseline_Group_Title"] = canonical.baseline_title
            row["Outcome_Group_ID"] = group_id
            row["Outcome_Group_Title"] = group_title
            row["EP_Name"] = outcome_title
            row["EP_type"] = outcome_type
            rows.append(row)
            used_groups.add(id(canonical))

    for group in alignment.groups:
        if id(group) in used_groups:
            continue
        alignment.ensure_arm_id(group, nct_id)
        row = {field: "" for field in GROUP_ARM_MAPPING_FIELDS}
        row["StudyID"] = nct_id
        row["Arm_ID"] = group.arm_id
        row["Arm_Title"] = group.best_title()
        row["Flow_Group_ID"] = group.flow_group_id
        row["Flow_Group_Title"] = group.flow_title
        row["Baseline_Group_ID"] = group.baseline_group_id
        row["Baseline_Group_Title"] = group.baseline_title
        rows.append(row)

    return rows


def filter_measures_for_group(measures: List[Dict[str, object]], group_id: str) -> List[Dict[str, object]]:
    gid = (group_id or "").strip()
    filtered: List[Dict[str, object]] = []
    for measure in measures:
        values = [
            value
            for value in (measure.get("values") or [])
            if isinstance(value, dict)
            and (value.get("group_id") == gid if gid else not value.get("group_id"))
        ]
        analyzed_items: List[Dict[str, object]] = []
        for analyzed in measure.get("analyzed") or []:
            counts = [
                count
                for count in analyzed.get("counts") or []
                if isinstance(count, dict)
                and (count.get("group_id") == gid if gid else not count.get("group_id"))
            ]
            if not counts:
                continue
            updated_analyzed = dict(analyzed)
            updated_analyzed["counts"] = counts
            analyzed_items.append(updated_analyzed)
        if not values and not analyzed_items:
            continue
        updated = dict(measure)
        if values:
            updated["values"] = values
        else:
            updated.pop("values", None)
        if analyzed_items:
            updated["analyzed"] = analyzed_items
        else:
            updated.pop("analyzed", None)
        filtered.append(updated)
    return filtered


def extract_r_arm_rows(root: ET.Element, fields: List[str], nct_id: str) -> Iterable[Dict[str, str]]:
    results = root.find("clinical_results")
    if results is None:
        return []

    arms = parse_arm_groups(root)
    arm_desc_map = {normalize_label(arm.get("label", "")): str(arm.get("description") or "").strip() for arm in arms}
    arm_type_candidates = [normalize_whitespace(str(arm.get("type") or "")) for arm in arms]
    unique_types = {value for value in arm_type_candidates if value}
    default_arm_type = unique_types.pop() if len(unique_types) == 1 else ""
    alignment = build_group_alignment(root, nct_id)

    interventions = parse_interventions(root)
    intervention_by_label: Dict[str, List[str]] = {}
    for iv in interventions:
        if not is_drug_like(str(iv.get("type") or "")):
            continue
        name = str(iv.get("name") or "").strip()
        if not name or is_placebo_like_name(name):
            continue
        for label in iv.get("labels") or []:
            key = normalize_label(label)
            if key:
                intervention_by_label.setdefault(key, []).append(name)

    baseline_groups = parse_group_list(results.find("baseline"))
    baseline_desc_by_id: Dict[str, str] = {}
    baseline_desc_by_title: Dict[str, str] = {}
    for group in baseline_groups:
        group_id = group.get("group_id") or ""
        title = group.get("title") or ""
        desc = group.get("description") or ""
        text = desc or title
        if group_id:
            baseline_desc_by_id[group_id] = text
        if title:
            baseline_desc_by_title[normalize_label(title)] = text

    outcomes = results.findall("outcome_list/outcome")
    rows: List[Dict[str, str]] = []
    for outcome in outcomes:
        outcome_title = xml_text(outcome, "title")
        outcome_type = outcome_type_label(xml_text(outcome, "type"))

        groups = parse_group_list(outcome)
        measures = parse_outcome_measures(outcome)

        measure_group_ids = collect_measure_group_ids(measures)

        group_map = {group.get("group_id") or f"title:{normalize_label(group.get('title', ''))}": group for group in groups}
        for group_id in sorted(measure_group_ids):
            if group_id not in group_map:
                group_map[group_id] = {"group_id": group_id, "title": "", "description": ""}

        if not group_map:
            group_map = {"overall": {"group_id": "", "title": "Overall", "description": ""}}

        outcome_cache: Dict[str, CanonicalGroup] = {}
        for group in group_map.values():
            row = {field: "" for field in fields}
            set_if_present(row, "StudyID", nct_id)

            group_id = (group.get("group_id") or "").strip()
            group_title = str(group.get("title") or "").strip()
            group_desc = str(group.get("description") or "").strip()

            canonical = None
            if group_id and group_id in outcome_cache:
                canonical = outcome_cache[group_id]
            else:
                canonical = alignment.match(group_id, group_title, group_desc)
                if canonical is None:
                    canonical = alignment.add_outcome_group(group_id, group_title, group_desc)
                alignment.ensure_arm_id(canonical, nct_id)
                if group_id:
                    outcome_cache[group_id] = canonical

            if canonical:
                if not group_title:
                    group_title = canonical.best_title()
                if not group_desc:
                    group_desc = canonical.best_desc()

            arm_id = canonical.arm_id if canonical else f"{nct_id}_A{alignment.next_fallback_index}"
            set_if_present(row, "Arm_ID", arm_id)
            set_if_present(row, "EP_Name", outcome_title)
            set_if_present(row, "EP_type", outcome_type)

            ep_pop = group_pop_text(group_title, group_desc)
            set_if_present(row, "EP_Pop", ep_pop)

            baseline_desc = ""
            if group_id and group_id in baseline_desc_by_id:
                baseline_desc = baseline_desc_by_id[group_id]
            elif group_title:
                baseline_desc = baseline_desc_by_title.get(normalize_label(group_title), "")
            set_if_present(row, "Baseline_Group_Desc", baseline_desc)

            arm_desc = arm_desc_map.get(normalize_label(group_title), "")
            if not arm_desc:
                arm_desc = group_desc
            set_if_present(row, "Arm_Description", arm_desc)

            group_type = ""
            if canonical and canonical.arm_type:
                group_type = canonical.arm_type
            if not group_type:
                group_type = default_arm_type
            set_if_present(row, "Group_Type", group_type)

            drug_names = intervention_by_label.get(normalize_label(group_title), [])
            set_if_present(row, "Drug_Name", join_values(drug_names))

            group_measures = filter_measures_for_group(measures, group_id)

            point_entries: List[Dict[str, object]] = []
            ci_entries: List[Dict[str, object]] = []
            analyzed_entries: List[Dict[str, object]] = []
            for measure in group_measures:
                measure_title = str(measure.get("measure_title") or "").strip()
                entry: Dict[str, object] = {}
                if measure_title:
                    entry["measure_title"] = measure_title
                if measure.get("units"):
                    entry["units"] = measure.get("units")
                if measure.get("param"):
                    entry["param"] = measure.get("param")
                if measure.get("dispersion"):
                    entry["dispersion"] = measure.get("dispersion")

                values = []
                ci_values = []
                for value in measure.get("values") or []:
                    if not isinstance(value, dict):
                        continue
                    value_entry = {}
                    category = str(value.get("category") or "").strip()
                    if category:
                        value_entry["category"] = category
                    if value.get("value"):
                        value_entry["value"] = value.get("value")
                    if value.get("spread"):
                        value_entry["spread"] = value.get("spread")
                    if value_entry:
                        values.append(value_entry)

                    ci_entry = {}
                    if category:
                        ci_entry["category"] = category
                    if value.get("lower"):
                        ci_entry["lower"] = value.get("lower")
                    if value.get("upper"):
                        ci_entry["upper"] = value.get("upper")
                    if value.get("spread"):
                        ci_entry["spread"] = value.get("spread")
                    if ci_entry:
                        ci_values.append(ci_entry)

                if values:
                    entry["values"] = values
                    point_entries.append(entry)

                if ci_values:
                    ci_entry = {"measure_title": measure_title} if measure_title else {}
                    ci_entry["values"] = ci_values
                    ci_entries.append(ci_entry)

                analyzed_list = []
                for analyzed in measure.get("analyzed") or []:
                    if not isinstance(analyzed, dict):
                        continue
                    analyzed_entry: Dict[str, object] = {}
                    if analyzed.get("units"):
                        analyzed_entry["units"] = analyzed.get("units")
                    if analyzed.get("scope"):
                        analyzed_entry["scope"] = analyzed.get("scope")
                    counts = []
                    for count in analyzed.get("counts") or []:
                        if not isinstance(count, dict):
                            continue
                        count_entry = {}
                        if count.get("value"):
                            count_entry["value"] = count.get("value")
                        if count.get("group_id"):
                            count_entry["group_id"] = count.get("group_id")
                        if count_entry:
                            counts.append(count_entry)
                    if counts:
                        analyzed_entry["counts"] = counts
                    if analyzed_entry:
                        analyzed_list.append(analyzed_entry)
                if analyzed_list:
                    analyzed_payload: Dict[str, object] = {}
                    if measure_title:
                        analyzed_payload["measure_title"] = measure_title
                    analyzed_payload["analyzed"] = analyzed_list
                    analyzed_entries.append(analyzed_payload)

            if point_entries:
                set_if_present(row, "EP_Point", json.dumps(point_entries, ensure_ascii=False))
            if ci_entries:
                set_if_present(row, "EP_CI", json.dumps(ci_entries, ensure_ascii=False))
            if analyzed_entries:
                set_if_present(row, "Analyzed_N", json.dumps(analyzed_entries, ensure_ascii=False))

            rows.append(row)

    return rows
