"""Citation and source-grounding deterministic scorer."""

from __future__ import annotations

import re
from typing import Any

from agent_gate.models import EvalCase, ProviderResponse, ScoreResult
from agent_gate.scorers.base import Scorer

_CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.-]{0,63})\]")
_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9.%-]*")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "will",
    "must",
    "into",
    "under",
    "within",
    "after",
    "before",
    "about",
    "their",
    "they",
    "you",
    "your",
    "our",
    "based",
    "according",
    "provided",
    "document",
    "documentation",
    "please",
}


class CitationGroundingScorer(Scorer):
    """Verifies inline citations map to real context documents and support stated facts."""

    name = "citation_grounding"
    critical = False

    def score(
        self,
        case: EvalCase,
        response: ProviderResponse,
        config: dict[str, Any],
    ) -> ScoreResult:
        text = response.text or ""
        valid_docs = {doc.doc_id: doc.content for doc in case.context}
        found_citations = _CITATION_PATTERN.findall(text)
        unique_cited = list(dict.fromkeys(found_citations))

        fabricated = [cid for cid in unique_cited if cid not in valid_docs]
        valid_cited = [cid for cid in unique_cited if cid in valid_docs]

        required_doc_ids = [str(d) for d in config.get("required_doc_ids", [])]
        min_citations = int(config.get("min_citations", 1 if required_doc_ids else 0))
        required_facts = [str(f) for f in config.get("required_facts", [])]
        min_lexical_overlap = float(config.get("min_lexical_overlap", 0.15))

        failures: list[str] = []
        checks_total = 0
        checks_passed = 0

        # 1. No fabricated citation IDs.
        checks_total += 1
        if fabricated:
            failures.append(f"Fabricated citation IDs: {', '.join(fabricated)}")
        else:
            checks_passed += 1

        # 2. Minimum valid citation count.
        if min_citations > 0:
            checks_total += 1
            if len(valid_cited) >= min_citations:
                checks_passed += 1
            else:
                failures.append(
                    f"Expected at least {min_citations} valid citation(s), found {len(valid_cited)}"
                )

        # 3. Specific required source documents cited.
        if required_doc_ids:
            missing_docs = [doc_id for doc_id in required_doc_ids if doc_id not in valid_cited]
            checks_total += 1
            if not missing_docs:
                checks_passed += 1
            else:
                failures.append(f"Missing required citation(s): {', '.join(missing_docs)}")

        # 4. Required grounded facts appear in both response and context.
        lower_text = text.lower()
        for fact in required_facts:
            checks_total += 1
            if fact.lower() in lower_text:
                checks_passed += 1
            else:
                failures.append(f"Missing grounded fact: '{fact}'")

        # 5. Lexical overlap against cited context documents.
        overlap_ratio = 1.0
        if valid_cited and min_lexical_overlap > 0:
            checks_total += 1
            cited_corpus = " ".join(valid_docs[cid] for cid in valid_cited)
            overlap_ratio = _lexical_overlap(text, cited_corpus)
            if overlap_ratio >= min_lexical_overlap:
                checks_passed += 1
            else:
                failures.append(
                    f"Low lexical grounding overlap ({overlap_ratio:.2f} < {min_lexical_overlap:.2f})"
                )

        score_val = round(checks_passed / checks_total, 4) if checks_total else 1.0
        passed = len(failures) == 0
        reason = (
            f"All {len(valid_cited)} citation(s) verified against context."
            if passed
            else "; ".join(failures)
        )
        return ScoreResult(
            name=self.name,
            passed=passed,
            score=score_val,
            reason=reason,
            critical=bool(config.get("critical", self.critical)),
            details={
                "valid_citations": valid_cited,
                "fabricated_citations": fabricated,
                "lexical_overlap": round(overlap_ratio, 4),
            },
        )


def _lexical_overlap(response_text: str, source_text: str) -> float:
    resp_tokens = {
        tok
        for tok in _TOKEN_PATTERN.findall(response_text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }
    if not resp_tokens:
        return 0.0
    src_tokens = {
        tok
        for tok in _TOKEN_PATTERN.findall(source_text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }
    if not src_tokens:
        return 0.0
    return len(resp_tokens & src_tokens) / len(resp_tokens)
