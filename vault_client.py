"""
vault_client.py — F13LD.vault Supabase REST client (read-only).

Thin Python wrapper around the f13ld_designs table on Supabase. Used by
train_synth.py's --from-vault mode to pull design records directly without
any local sweep-file management. Family-agnostic: filters are passed in,
nothing about TPMS is hardcoded.

Setup
-----
For the canonical F13LD.vault project: no setup needed. The Supabase URL and
anon key are baked in as defaults (the same values F13LD.synth's HTML uses,
since the anon key is public-safe by design).

To point at a different project, either pass arguments to the constructor:
    vc = VaultClient(url="https://other.supabase.co", anon_key="...")
or set environment variables (these override the defaults):
    VAULT_SUPABASE_URL, VAULT_SUPABASE_ANON_KEY

Programmatic use
----------------
    from vault_client import VaultClient
    vc = VaultClient()
    designs = vc.fetch_designs(family='tpms', valid_only=True)
    # designs is List[dict], one dict per row, mirroring all columns

Dependencies: requests (only). Tested against Supabase PostgREST v12+.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# --------------------------------------------------------------------------
# Default credentials. The Supabase anon key is public-safe by design and
# already embedded in F13LD.synth's HTML — duplicating it here means the
# trainer just works for the canonical F13LD.vault project with no setup.
# Override via env vars or constructor args if pointing at a different project.
# --------------------------------------------------------------------------
DEFAULT_VAULT_URL = "https://axinljpecycnvfncyhfs.supabase.co"
DEFAULT_VAULT_ANON_KEY = "sb_publishable_DAlrNLqbUZiwkaA6wPSMIw_YUNY85LX"

ENV_URL_PREFERRED = "VAULT_SUPABASE_URL"
ENV_KEY_PREFERRED = "VAULT_SUPABASE_ANON_KEY"
ENV_URL_FALLBACK = "SUPABASE_URL"
ENV_KEY_FALLBACK = "SUPABASE_ANON_KEY"


def _resolve_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class VaultClient:
    """Read-only client for F13LD.vault. One instance per process is fine."""

    DEFAULT_TABLE = "f13ld_designs"
    PAGE_SIZE = 1000           # Supabase per-request cap
    REQUEST_TIMEOUT = 30       # seconds, per HTTP call
    MAX_RETRIES = 3            # transient-error retry budget per page

    def __init__(
        self,
        url: Optional[str] = None,
        anon_key: Optional[str] = None,
        table: Optional[str] = None,
    ) -> None:
        # Resolution order: explicit arg → preferred env var → fallback env var → built-in default.
        url = url or _resolve_env(ENV_URL_PREFERRED, ENV_URL_FALLBACK) or DEFAULT_VAULT_URL
        key = anon_key or _resolve_env(ENV_KEY_PREFERRED, ENV_KEY_FALLBACK) or DEFAULT_VAULT_ANON_KEY
        self.url = url.rstrip("/")
        self.anon_key = key
        self.table = table or self.DEFAULT_TABLE

    # -----------------------------------------------------------------
    # Internal HTTP helpers
    # -----------------------------------------------------------------
    @property
    def _base_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {self.anon_key}",
            "Accept": "application/json",
        }

    def _endpoint(self) -> str:
        return f"{self.url}/rest/v1/{self.table}"

    def _build_filter_params(
        self,
        family: Optional[str],
        valid_only: bool,
        exclude_degenerate: bool,
        since: Optional[Any],
    ) -> List[Tuple[str, str]]:
        """Build PostgREST filter params (list of (key, value) tuples)."""
        params: List[Tuple[str, str]] = []
        if family is not None:
            params.append(("family", f"eq.{family}"))
        if valid_only:
            # neq excludes NULLs in PostgREST too, which is what we want —
            # rows without a recorded validity are not safe training data.
            params.append(("solver_validity", "neq.invalid"))
        if exclude_degenerate:
            # `not.is.true` covers both false and NULL; robust to either schema.
            params.append(("degenerate", "not.is.true"))
        if since is not None:
            iso = since.isoformat() if isinstance(since, datetime) else str(since)
            params.append(("created_at", f"gte.{iso}"))
        return params

    @staticmethod
    def _parse_total(content_range: Optional[str]) -> Optional[int]:
        """Extract total row count from a 'Content-Range: a-b/total' header."""
        if not content_range or "/" not in content_range:
            return None
        tail = content_range.split("/", 1)[1].strip()
        if tail in ("", "*"):
            return None
        try:
            return int(tail)
        except ValueError:
            return None

    def _get_with_retry(
        self,
        params: List[Tuple[str, str]],
        headers: Dict[str, str],
    ) -> requests.Response:
        """GET with simple retry-on-transient-error. Raises on terminal failure."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                r = requests.get(
                    self._endpoint(),
                    headers=headers,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT,
                )
                if r.status_code in (429, 502, 503, 504) and attempt < self.MAX_RETRIES:
                    time.sleep(0.5 * attempt)
                    continue
                r.raise_for_status()
                return r
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < self.MAX_RETRIES:
                    time.sleep(0.5 * attempt)
                    continue
                raise
        # Defensive — loop should always either return or raise.
        if last_err:
            raise last_err
        raise RuntimeError("vault_client: unreachable retry exit")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def count_designs(
        self,
        family: Optional[str] = None,
        valid_only: bool = True,
        exclude_degenerate: bool = True,
        since: Optional[Any] = None,
    ) -> int:
        """Return number of rows matching filters, without fetching them."""
        params = [("select", "created_at"), ("limit", "1")]
        params.extend(self._build_filter_params(family, valid_only, exclude_degenerate, since))
        headers = dict(self._base_headers)
        headers["Prefer"] = "count=exact"
        r = self._get_with_retry(params, headers)
        return self._parse_total(r.headers.get("Content-Range")) or 0

    def fetch_designs(
        self,
        family: Optional[str] = None,
        valid_only: bool = True,
        exclude_degenerate: bool = True,
        since: Optional[Any] = None,
        limit: Optional[int] = None,
        select: str = "*",
        order_by: str = "created_at.asc",
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch design rows from f13ld_designs.

        Args:
            family: filter to a single family (e.g. 'tpms'). None = all.
            valid_only: drop rows where solver_validity is 'invalid' or NULL.
            exclude_degenerate: drop rows where degenerate is true.
            since: only rows with created_at >= this datetime/ISO string.
            limit: cap on total rows returned. None = all.
            select: PostgREST select clause; '*' returns all columns.
            order_by: PostgREST order clause. ASC by date is recommended for
                deterministic, replay-friendly training (newest data appears
                last in the list).
            verbose: print pagination progress to stderr.

        Returns:
            List of row dicts. Each dict has keys matching the columns
            requested in `select`.
        """
        filters = self._build_filter_params(family, valid_only, exclude_degenerate, since)
        rows: List[Dict[str, Any]] = []
        offset = 0
        total: Optional[int] = None

        while True:
            remaining = (limit - len(rows)) if limit is not None else self.PAGE_SIZE
            page_size = min(self.PAGE_SIZE, remaining)
            if page_size <= 0:
                break

            params: List[Tuple[str, str]] = [("select", select), ("order", order_by)]
            params.extend(filters)
            params.append(("offset", str(offset)))
            params.append(("limit", str(page_size)))

            # Request exact total only on the first page — saves DB work after.
            headers = dict(self._base_headers)
            if offset == 0:
                headers["Prefer"] = "count=exact"

            r = self._get_with_retry(params, headers)
            page = r.json()
            rows.extend(page)

            if offset == 0:
                total = self._parse_total(r.headers.get("Content-Range"))

            if verbose:
                if total is not None:
                    print(
                        f"[vault_client] fetched {len(rows)}/{total} rows",
                        file=sys.stderr,
                    )
                else:
                    print(f"[vault_client] fetched {len(rows)} rows", file=sys.stderr)

            # Stop if the page came back short (we hit the end) or we've hit limit.
            if len(page) < page_size:
                break
            if limit is not None and len(rows) >= limit:
                break
            offset += len(page)

        return rows


# --------------------------------------------------------------------------
# CLI for quick verification — `python vault_client.py --help`
# --------------------------------------------------------------------------
def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="F13LD.vault read-only REST client. Quick connectivity and schema checks."
    )
    p.add_argument("--family", default=None, help="filter by family (e.g. tpms)")
    p.add_argument(
        "--include-invalid",
        action="store_true",
        help="don't filter out solver_validity=invalid (default: filter)",
    )
    p.add_argument(
        "--include-degenerate",
        action="store_true",
        help="don't filter out degenerate=true (default: filter)",
    )
    p.add_argument("--since", default=None, help="ISO datetime, e.g. 2025-01-01")
    p.add_argument("--limit", type=int, default=None, help="cap on rows fetched")
    p.add_argument("--select", default="*", help="PostgREST select clause (default: *)")
    p.add_argument(
        "--count",
        action="store_true",
        help="just print the matching row count and exit",
    )
    p.add_argument(
        "--out",
        default=None,
        help="write fetched rows to a JSON file (else prints schema summary)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress pagination progress messages",
    )
    args = p.parse_args(argv)

    try:
        vc = VaultClient()
    except ValueError as e:
        print(f"vault_client: {e}", file=sys.stderr)
        return 2

    if args.count:
        n = vc.count_designs(
            family=args.family,
            valid_only=not args.include_invalid,
            exclude_degenerate=not args.include_degenerate,
            since=args.since,
        )
        print(n)
        return 0

    rows = vc.fetch_designs(
        family=args.family,
        valid_only=not args.include_invalid,
        exclude_degenerate=not args.include_degenerate,
        since=args.since,
        limit=args.limit,
        select=args.select,
        verbose=not args.quiet,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"[vault_client] wrote {len(rows)} rows -> {args.out}")
        return 0

    # No output file requested: print a schema summary so the user can
    # see what columns exist on the rows they just fetched.
    print(f"[vault_client] {len(rows)} rows fetched")
    if not rows:
        return 0
    sample = rows[0]
    print(f"[vault_client] sample row has {len(sample)} columns:")
    for k in sorted(sample.keys()):
        v = sample[k]
        repr_v = repr(v)
        if len(repr_v) > 80:
            repr_v = repr_v[:77] + "..."
        print(f"  {k:32s}  {repr_v}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
