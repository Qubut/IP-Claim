"""Parse the HUPD ``claims`` blob into typed :class:`Claim` value objects.

HUPD stores all claims of a patent as a single string, with each claim
prefixed by its number (e.g. ``"1. A method ... 2. The method of claim 1, ..."``).
We split on the leading-number boundary, normalize whitespace, and detect
dependency by scanning for ``"claim N"`` (case-insensitive) inside the text.

Unconventional numbering (``(1)``, ``[1]``, ``1)``) falls back to one
independent claim ``number=1`` rather than failing the parse.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import starmap
from operator import attrgetter
from typing import Final

from more_itertools import pairwise, unique_everseen
from returns.maybe import Maybe
from returns.methods import partition
from returns.pipeline import flow, is_successful

from ip_claim.ingestion.models import Claim

# A claim header boundary inside a HUPD blob.
#
# Real HUPD ``claims`` strings are emitted as a single line (no newlines)
# with claims concatenated like ``"1. A device. 2. The device of claim 1...
# 12. (canceled) 13. (canceled) 14. An apparatus..."``. We therefore anchor
# on either:
#   * the start of the blob (``\A``), OR
#   * a sentence-ending punctuation followed by whitespace
#     (``[.):\]]\s``) — covers ``...end. N.`` and ``(canceled) N.``.
#
# A trailing lookahead ``(?=[A-Z(])`` confirms the next claim body starts
# with an uppercase letter or an opening parenthesis (the ``(canceled)``
# marker), avoiding false matches on sub-references like
# ``"as recited in claim 12, the method...``.
_CLAIM_HEADER_RE: Final = re.compile(
    r'(?:\A|(?<=[.):\]]\s))([1-9]\d{0,3})\.\s+(?=[A-Z(])',
    re.MULTILINE,
)
_DEPENDENCY_RE: Final = re.compile(
    r'\b(?:of|according to|as in|as set forth in|as recited in|as defined in)'
    r'\s+claim\s+(\d{1,4})\b',
    re.IGNORECASE,
)
_WHITESPACE_RE: Final = re.compile(r'\s+')


def _normalize(text: str) -> str:
    """Collapse runs of whitespace into a single space."""
    return _WHITESPACE_RE.sub(' ', text).strip()


def _detect_parent(text: str) -> Maybe[int]:
    """Return ``Some(parent_number)`` iff the body cites ``claim N``."""
    return Maybe.from_optional(_DEPENDENCY_RE.search(text)).map(lambda m: int(m.group(1)))


def _build_claim(number: int, raw_body: str) -> Maybe[Claim]:
    """Normalize the body, drop empty slices, then assemble a Claim."""

    def as_claim(text: str) -> Claim:
        parent = _detect_parent(text)
        return Claim(
            number=number,
            text=text,
            is_independent=not is_successful(parent),
            parent_number=parent.value_or(None),
        )

    return flow(
        raw_body,
        _normalize,
        lambda body: Maybe.from_optional(body or None).map(as_claim),
    )


# Sentinel match-like end-of-blob marker; exposes ``.start()`` and lets us
# treat the last header-to-EOF slice with the same shape as inner slices.
class _EndSentinel:
    """End-of-blob sentinel: only ``start`` matters for slice arithmetic."""

    __slots__ = ('_pos',)

    def __init__(self, pos: int) -> None:
        self._pos = pos

    def start(self) -> int:
        """Return the absolute end-of-blob position (exclusive)."""
        return self._pos


def _pair_to_slice(curr: re.Match[str], nxt: re.Match[str] | _EndSentinel) -> tuple[int, str]:
    """Convert a ``(header, next_boundary)`` pair into ``(claim_number, body)``."""
    return (int(curr.group(1)), curr.string[curr.end() : nxt.start()])


def _slice_bodies(blob: str) -> Iterable[tuple[int, str]]:
    """Stream ``(claim_number, raw_body)`` pairs in source order.

    Headers are paired with the next boundary (or end of blob). Falls back
    to ``[(1, blob)]`` when no header is found.
    """
    headers = tuple(_CLAIM_HEADER_RE.finditer(blob))
    if not headers:
        return ((1, blob),)
    boundaries: tuple[re.Match[str] | _EndSentinel, ...] = (
        *headers,
        _EndSentinel(len(blob)),
    )
    return starmap(_pair_to_slice, pairwise(boundaries))


def parse_claims(blob: str) -> tuple[Claim, ...]:
    """Parse a HUPD claims blob into ordered :class:`Claim` value objects.

    Returns the empty tuple for empty/whitespace-only input.

    Slices become optional claims; empty bodies drop out; first-wins
    de-duplication keeps a stable number order.
    """
    if not blob or not blob.strip():
        return ()
    kept, _ = partition(starmap(_build_claim, _slice_bodies(blob)))
    return tuple(unique_everseen(kept, key=attrgetter('number')))
