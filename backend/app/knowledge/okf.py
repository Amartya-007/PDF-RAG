from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.core.hashing import stable_id

# Link matching regex is fine to keep, but frontmatter regex is removed for speed.
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")


@dataclass(frozen=True)
class OkfConcept:
    concept_id: str
    title: str
    slug: str
    text: str
    source_chunk_ids: list[str]
    verification_status: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    path: str | None = None


@dataclass(frozen=True)
class OkfValidationIssue:
    path: str
    severity: str
    message: str


@dataclass(frozen=True)
class ParsedOkfFile:
    path: Path
    metadata: dict[str, object]
    body: str
    links: list[str]


class OkfValidationError(ValueError):
    def __init__(self, issues: list[OkfValidationIssue]) -> None:
        self.issues = issues
        joined = "; ".join(issue.message for issue in issues)
        super().__init__(joined)


def parse_okf_markdown(path: Path) -> ParsedOkfFile:
    """Reads a file and extracts frontmatter using fast string splitting."""
    text = path.read_text(encoding="utf-8")
    
    # Fast path: strictly check if it starts with standard YAML frontmatter markers
    if text.startswith("---\n"):
        # split by '---' at most 2 times. 
        # parts[0] is empty, parts[1] is frontmatter, parts[2] is body.
        parts = text.split("---\n", 2)
        if len(parts) >= 3:
            raw_frontmatter = parts[1]
            body = parts[2]
            metadata = parse_simple_yaml(raw_frontmatter)
            return ParsedOkfFile(path=path, metadata=metadata, body=body, links=_extract_links(body))
            
    # Fallback if no frontmatter exists
    return ParsedOkfFile(path=path, metadata={}, body=text, links=_extract_links(text))


def parse_simple_yaml(raw: str) -> dict[str, object]:
    """Lightweight hand-rolled YAML parser."""
    result: dict[str, object] = {}
    current_key: str | None = None
    current_list: list[object] | None = None
    current_map: dict[str, object] | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or line.lstrip().startswith("#"):
            continue
            
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            current_map = None
            value = value.strip()
            
            if not value:
                current_list = []
                result[current_key] = current_list
            else:
                current_list = None
                result[current_key] = _parse_scalar(value)
            continue

        if current_key is None or current_list is None:
            continue

        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if ":" in item and not item.startswith("["):
                key, value = item.split(":", 1)
                current_map = {key.strip(): _parse_scalar(value.strip())}
                current_list.append(current_map)
            else:
                current_map = None
                current_list.append(_parse_scalar(item))
        elif current_map is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_map[key.strip()] = _parse_scalar(value.strip())

    return result


def render_simple_yaml(metadata: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in metadata.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    first = True
                    for item_key, item_value in item.items():
                        prefix = "  - " if first else "    "
                        lines.append(f"{prefix}{item_key}: {_format_scalar(item_value)}")
                        first = False
                else:
                    lines.append(f"  - {_format_scalar(item)}")
        else:
            lines.append(f"{key}: {_format_scalar(value)}")
    return "\n".join(lines)


def validate_okf_bundle(root: Path) -> list[OkfValidationIssue]:
    """Validates an entire OKF directory."""
    files = [path for path in root.rglob("*.md") if path.name.lower() != "index.md"]
    parsed_files = [parse_okf_markdown(path) for path in files]
    return _validate_parsed_files(parsed_files, root)


def _validate_parsed_files(parsed_files: list[ParsedOkfFile], root: Path) -> list[OkfValidationIssue]:
    """Core validation logic separated so it can reuse in-memory parsed files."""
    issues: list[OkfValidationIssue] = []
    
    if not parsed_files:
        return [_issue(str(root), "error", "No Markdown concept files found.")]

    known_relative_paths = {p.path.relative_to(root).as_posix() for p in parsed_files}
    known_names = {p.path.name for p in parsed_files}

    for parsed in parsed_files:
        relative = parsed.path.relative_to(root).as_posix()
        metadata = parsed.metadata
        concept_type = metadata.get("type")
        
        if not concept_type:
            issues.append(_issue(relative, "error", "Missing required OKF frontmatter field: type."))
            
        if concept_type == "concept":
            if not metadata.get("title"):
                issues.append(_issue(relative, "error", "Concept files must include title."))
            if not metadata.get("id"):
                issues.append(_issue(relative, "error", "Concept files must include id."))
                
        if not parsed.body.strip():
            issues.append(_issue(relative, "warning", "OKF file body is empty."))
            
        for field_name in ("tags", "aliases", "related", "depends_on"):
            value = metadata.get(field_name, [])
            if value and not isinstance(value, list):
                issues.append(_issue(relative, "error", f"{field_name} must be a list."))
                
        for link in parsed.links:
            normalized = _normalize_link(relative, link)
            if normalized not in known_relative_paths and Path(link).name not in known_names:
                issues.append(_issue(relative, "warning", f"Markdown link target not found: {link}."))

    return issues


def import_okf_bundle(source_root: Path, target_root: Path, *, fail_on_error: bool = True) -> list[OkfConcept]:
    """Imports an OKF bundle, reusing parsed files to prevent double Disk I/O."""
    files = [path for path in source_root.rglob("*.md") if path.name.lower() != "index.md"]
    
    # 1. Parse all files into memory exactly once
    parsed_files = [parse_okf_markdown(path) for path in files]
    
    # 2. Validate the in-memory files
    issues = _validate_parsed_files(parsed_files, source_root)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors and fail_on_error:
        raise OkfValidationError(errors)

    # 3. Import and copy
    concepts: list[OkfConcept] = []
    for parsed in parsed_files:
        if parsed.metadata.get("type") != "concept":
            continue
            
        relative = parsed.path.relative_to(source_root)
        target_path = target_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parsed.path, target_path)
        
        concepts.append(concept_from_parsed(target_path, parsed))
        
    return concepts


def concept_from_parsed(path: Path, parsed: ParsedOkfFile | None = None) -> OkfConcept:
    parsed = parsed or parse_okf_markdown(path)
    metadata = parsed.metadata
    title = str(metadata.get("title") or path.stem.replace("-", " ").title())
    slug = str(metadata.get("slug") or path.stem)
    concept_id = str(metadata.get("id") or stable_id("concept", slug, parsed.body))
    
    source_chunk_ids = _string_list(metadata.get("source_chunk_ids"))
    if not source_chunk_ids:
        source_chunk_ids = [
            str(item.get("chunk_id"))
            for item in _dict_list(metadata.get("source_chunks"))
            if item.get("chunk_id")
        ]
        
    text = f"# {title}\n\n{parsed.body.strip()}".strip()
    return OkfConcept(
        concept_id=concept_id,
        title=title,
        slug=slug,
        text=text,
        source_chunk_ids=source_chunk_ids,
        verification_status=str(metadata.get("verification_status") or "unverified"),
        aliases=_string_list(metadata.get("aliases")),
        tags=_string_list(metadata.get("tags")),
        related=_string_list(metadata.get("related")),
        depends_on=_string_list(metadata.get("depends_on")),
        path=str(path),
    )


def _extract_links(text: str) -> list[str]:
    return [match.group(1) for match in LINK_RE.finditer(text)]


def _normalize_link(source_relative: str, link: str) -> str:
    base = Path(source_relative).parent
    return (base / link).as_posix()


def _issue(path: str, severity: str, message: str) -> OkfValidationIssue:
    return OkfValidationIssue(path=path, severity=severity, message=message)


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _format_scalar(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if not text or any(char in text for char in (":", "#", "[", "]", "{", "}")):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _string_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if not isinstance(item, dict)]
    return [str(value)]


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]