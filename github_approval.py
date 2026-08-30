"""Authoritative GitHub approval verification for sealed scientific runs."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_scientific_approval(
    comment_url: str,
    attempt_id: str,
    gate_receipt: Path,
    source_git_commit: str,
    contract: dict[str, Any],
    urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    """Fetch and verify the exact approval comment against sealed inputs."""
    repository = str(contract["github_repository"])
    prefixes = tuple(
        f"https://github.com/{repository}/{kind}/" for kind in ("issues", "pull")
    )
    if not comment_url.startswith(prefixes) or "#issuecomment-" not in comment_url:
        raise RuntimeError("scientific approval must be an exact GitHub issue comment")
    comment_id = comment_url.rsplit("#issuecomment-", 1)[1]
    if re.fullmatch(r"[1-9][0-9]*", comment_id) is None:
        raise RuntimeError("scientific approval comment ID is invalid")
    api_url = f"https://api.github.com/repos/{repository}/issues/comments/{comment_id}"
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "TREAT-MMTB"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            comment = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError("GitHub approval verification is unavailable") from error
    if not isinstance(comment, dict):
        raise TypeError("GitHub approval response is invalid")
    marker_line = f"<!-- {contract['body_marker']} -->"
    body = str(comment.get("body", ""))
    if marker_line not in body:
        raise RuntimeError("GitHub approval marker is missing")
    try:
        approval = json.loads(body.split(marker_line, 1)[1].strip())
    except json.JSONDecodeError as error:
        raise RuntimeError("GitHub approval payload is invalid") from error
    gate_id = f"{attempt_id}-resource-gate"
    login = comment.get("user", {}).get("login")
    if (
        not isinstance(approval, dict)
        or approval.get("schema_version") != 1
        or approval.get("status") != "approved"
        or approval.get("attempt_id") != attempt_id
        or approval.get("gate_attempt_id") != gate_id
        or approval.get("source_git_commit") != source_git_commit
        or re.fullmatch(r"[0-9a-f]{40}", source_git_commit) is None
        or approval.get("resource_gate_receipt_sha256") != _sha256_file(gate_receipt)
        or login not in contract["allowed_reviewers"]
        or comment.get("author_association")
        not in contract["required_author_associations"]
        or approval.get("reviewed_by") != login
        or approval.get("review_url") != comment_url
        or approval.get("external_final_test_untouched") is not True
    ):
        raise RuntimeError("scientific soak approval differs from reviewed gate")
    return {
        **approval,
        "approval_sha256": _canonical_sha256(comment),
        "github_api_url": api_url,
    }
