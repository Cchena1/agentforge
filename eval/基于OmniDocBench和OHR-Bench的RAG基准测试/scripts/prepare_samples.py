from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT / "state" / "benchmark-200p"
OMNI_ANNOTATIONS = ROOT / "OmniDocBench.json"
OHR_QAS = ROOT / "ohr_qas_v2.json"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AgentForge-Benchmark-Runner/0.3"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, *, attempts: int = 5) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        response = SESSION.get(url, timeout=180)
        if response.status_code == 200:
            return response.json()
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == attempts:
            response.raise_for_status()
        time.sleep(4 * attempt)
    raise AssertionError("unreachable")


def download(url: str, target: Path, *, attempts: int = 5) -> None:
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        response = SESSION.get(url, timeout=300)
        if response.status_code == 200:
            target.write_bytes(response.content)
            return
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == attempts:
            response.raise_for_status()
        time.sleep(4 * attempt)


def attr(item: dict[str, Any]) -> dict[str, Any]:
    return item["page_info"]["page_attribute"]


def includes_issue(item: dict[str, Any], token: str) -> bool:
    return token in attr(item).get("special_issue", [])


def omni_gt_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    ordered = sorted(
        (det for det in item.get("layout_dets", []) if not det.get("ignore", False)),
        key=lambda det: (det.get("order") is None, det.get("order", 10**9), det.get("anno_id", "")),
    )
    for det in ordered:
        value = det.get("text") or det.get("latex") or det.get("html") or ""
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


STRATA: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
    ("english_academic_double_column", lambda x: attr(x).get("data_source") == "academic_literature" and attr(x).get("language") == "english" and attr(x).get("layout") == "double_column"),
    ("chinese_newspaper_complex_layout", lambda x: attr(x).get("data_source") == "newspaper" and attr(x).get("language") == "simplified_chinese" and attr(x).get("layout") == "other_layout"),
    ("english_newspaper_three_column", lambda x: attr(x).get("data_source") == "newspaper" and attr(x).get("language") == "english" and attr(x).get("layout") == "three_column"),
    ("chinese_table_hard", lambda x: attr(x).get("language") == "simplified_chinese" and attr(x).get("subset") == "table_hard"),
    ("english_table_hard", lambda x: attr(x).get("language") == "english" and attr(x).get("subset") == "table_hard"),
    ("english_academic_equation_hard", lambda x: attr(x).get("data_source") == "academic_literature" and attr(x).get("language") == "english" and attr(x).get("subset") == "equation_hard"),
    ("chinese_ppt_color_background", lambda x: attr(x).get("data_source") == "PPT2PDF" and attr(x).get("language") == "simplified_chinese" and includes_issue(x, "colorful_backgroud")),
    ("mixed_language_notes", lambda x: attr(x).get("data_source") == "note" and attr(x).get("language") == "en_ch_mixed"),
    ("english_exam_multicolumn", lambda x: attr(x).get("data_source") == "exam_paper" and attr(x).get("language") == "english" and attr(x).get("layout") in {"double_column", "1andmore_column"}),
    ("english_book_equation_hard", lambda x: attr(x).get("data_source") == "book" and attr(x).get("language") == "english" and attr(x).get("subset") == "equation_hard"),
]


def prepare_omni() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(OMNI_ANNOTATIONS.read_text(encoding="utf-8"))
    used: set[str] = set()
    image_dir = ROOT / "omni" / "images"
    pdf_dir = ROOT / "omni" / "pdfs"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []

    for stratum, predicate in STRATA:
        candidates = sorted(
            (item for item in data if predicate(item)),
            key=lambda item: item["page_info"]["image_path"],
        )
        selected_in_stratum = 0
        for item in candidates:
            image_name = item["page_info"]["image_path"]
            if image_name in used:
                continue
            image_path = image_dir / image_name
            url = f"https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/images/{image_name}?download=true"
            try:
                download(url, image_path)
            except requests.HTTPError as exc:
                unavailable.append({"stratum": stratum, "image_name": image_name, "error": str(exc)})
                continue
            used.add(image_name)
            selected_in_stratum += 1
            index = len(manifest) + 1
            pdf_path = pdf_dir / f"omni_{index:03d}.pdf"
            if not pdf_path.exists():
                with Image.open(image_path) as image:
                    converted = image.convert("RGB")
                    converted.save(pdf_path, "PDF", resolution=150.0)
            page_attr = attr(item)
            gt_text = omni_gt_text(item)
            manifest.append(
                {
                    "sample_id": f"omni-{index:03d}",
                    "benchmark": "OmniDocBench",
                    "stratum": stratum,
                    "image_name": image_name,
                    "image_path": str(image_path.relative_to(ROOT)),
                    "pdf_path": str(pdf_path.relative_to(ROOT)),
                    "image_sha256": sha256(image_path),
                    "pdf_sha256": sha256(pdf_path),
                    "width": item["page_info"]["width"],
                    "height": item["page_info"]["height"],
                    "page_attribute": page_attr,
                    "ground_truth_text": gt_text,
                    "ground_truth_chars": len(gt_text),
                    "ground_truth_blocks": sum(1 for det in item.get("layout_dets", []) if not det.get("ignore", False)),
                }
            )
            print(f"Omni {index:02d}/50 {stratum} {image_name}", flush=True)
            if selected_in_stratum == 5:
                break
        if selected_in_stratum != 5:
            raise RuntimeError(f"stratum {stratum} has only {selected_in_stratum} downloadable pages")

    (ROOT / "omni_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "omni_unavailable.json").write_text(json.dumps(unavailable, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


OHR_QUOTAS = {
    "academic": 22,
    "administration": 22,
    "finance": 22,
    "law": 21,
    "manual": 21,
    "news": 21,
    "textbook": 21,
}
OHR_OFFSETS = {
    "academic": 0,
    "administration": 1250,
    "finance": 2500,
    "law": 4500,
    "manual": 5750,
    "news": 7400,
    "textbook": 8000,
}


def qa_pages(qas: list[dict[str, Any]]) -> tuple[set[tuple[str, int]], dict[tuple[str, int], list[dict[str, Any]]]]:
    pairs: set[tuple[str, int]] = set()
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for qa in qas:
        raw_pages = qa.get("evidence_page_no", [])
        pages = raw_pages if isinstance(raw_pages, list) else [raw_pages]
        for raw_page in pages:
            pair = (qa["doc_name"], int(raw_page))
            pairs.add(pair)
            by_page.setdefault(pair, []).append(qa)
    return pairs, by_page


def fetch_ohr_window(domain: str, offset: int, length: int = 100) -> list[dict[str, Any]]:
    cache = ROOT / f"ohr_window_{domain}_{offset}_{length}.json"
    if not cache.exists():
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=opendatalab%2FOHR-Bench&config=default&split=train&offset={offset}&length={length}"
        )
        payload = get_json(url)
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        time.sleep(3)
    return json.loads(cache.read_text(encoding="utf-8"))["rows"]


def prepare_ohr() -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = json.loads(OHR_QAS.read_text(encoding="utf-8"))
    labeled_pairs, by_page = qa_pages(qas)
    selected: list[dict[str, Any]] = []
    for domain, quota in OHR_QUOTAS.items():
        rows = fetch_ohr_window(domain, OHR_OFFSETS[domain])
        domain_rows = [entry for entry in rows if entry["row"]["domain"] == domain]
        labeled = [entry for entry in domain_rows if (entry["row"]["doc_name"], int(entry["row"]["page_idx"])) in labeled_pairs]
        unlabeled = [entry for entry in domain_rows if entry not in labeled]
        chosen = (labeled + unlabeled)[:quota]
        if len(chosen) != quota:
            raise RuntimeError(f"domain {domain} has only {len(chosen)} rows in window")
        for entry in chosen:
            row = entry["row"]
            pair = (row["doc_name"], int(row["page_idx"]))
            selected.append(
                {
                    "sample_id": f"ohr-{len(selected)+1:03d}",
                    "benchmark": "OHR-Bench",
                    "row_idx": entry["row_idx"],
                    "domain": row["domain"],
                    "doc_name": row["doc_name"],
                    "page_idx": row["page_idx"],
                    "gt_text": row["gt_text"],
                    "formatting_noise_moderate": row["formatting_noise_moderate"],
                    "semantic_noise_MinerU_moderate": row["semantic_noise_MinerU_moderate"],
                    "qa_ids": [qa["ID"] for qa in by_page.get(pair, [])],
                }
            )
        print(f"OHR {domain}: {len(chosen)} pages, {sum(bool(x['qa_ids']) for x in selected if x['domain'] == domain)} QA-labeled", flush=True)
    if len(selected) != 150:
        raise RuntimeError(f"expected 150 OHR pages, got {len(selected)}")
    (ROOT / "ohr_manifest.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


if __name__ == "__main__":
    omni = prepare_omni()
    ohr = prepare_ohr()
    summary = {
        "schema_version": 1,
        "selection_rule": "sorted deterministic stratified sample; no random seed",
        "unique_pages": len(omni) + len(ohr),
        "benchmarks": {"OmniDocBench": len(omni), "OHR-Bench": len(ohr)},
        "omni_strata": Counter(item["stratum"] for item in omni),
        "ohr_domains": Counter(item["domain"] for item in ohr),
        "ohr_qa_labeled_pages": sum(bool(item["qa_ids"]) for item in ohr),
        "source_hashes": {
            "OmniDocBench.json": sha256(OMNI_ANNOTATIONS),
            "ohr_qas_v2.json": sha256(OHR_QAS),
        },
    }
    (ROOT / "sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
