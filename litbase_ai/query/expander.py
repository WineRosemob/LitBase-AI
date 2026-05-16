from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any

from litbase_ai.models import ExpandedQuery
from litbase_ai.query.translator import QueryTranslator
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


class QueryExpander:
    """Expand a user topic into multilingual, multi-style query variants."""

    DEFAULT_RELATED_TERMS = [
        "integrated assessment model",
        "energy system model",
        "scenario analysis",
        "decarbonization pathway",
        "climate policy",
        "power system planning",
    ]

    def __init__(self, llm_scorer=None):
        self.llm_scorer = llm_scorer
        self.translator = QueryTranslator()

    def expand(self, topic: str) -> ExpandedQuery:
        """Build expanded query with rule-based and optional LLM enrichment."""
        expanded = self._rule_based_expand(topic)
        if self._llm_available():
            llm_update = self._llm_expand(topic)
            if llm_update:
                expanded = self._merge_llm_update(expanded, llm_update)
        expanded.phrase_queries = self._build_phrase_queries(expanded)
        expanded.loose_queries = self._build_loose_queries(expanded)
        expanded.boolean_queries = self._build_boolean_queries(expanded)
        return expanded

    def _rule_based_expand(self, topic: str) -> ExpandedQuery:
        detected = self.translator.detect_language(topic)
        english_topic = topic if detected in {"en", "mixed"} else self.translator.to_english(topic)
        chinese_topic = topic if detected in {"zh", "mixed"} else self.translator.to_chinese(topic)

        en_keywords = self._extract_english_keywords(english_topic)
        zh_keywords = self._extract_chinese_keywords(chinese_topic)

        for known_en, known_zh in QueryTranslator.EN_TO_ZH.items():
            topic_lower = topic.lower()
            if known_en in topic_lower and known_en not in [x.lower() for x in en_keywords]:
                en_keywords.append(known_en)
            if known_zh in topic and known_zh not in zh_keywords:
                zh_keywords.append(known_zh)

        synonyms = self._build_synonyms(en_keywords, zh_keywords)
        related_terms = self._build_related_terms(en_keywords, zh_keywords)

        expanded = ExpandedQuery(
            original_topic=topic,
            detected_language=detected,
            english_topic=english_topic,
            chinese_topic=chinese_topic,
            english_keywords=self._dedupe(en_keywords),
            chinese_keywords=self._dedupe(zh_keywords),
            synonyms=self._dedupe(synonyms),
            related_terms=self._dedupe(related_terms),
        )
        return expanded

    def _build_english_queries(self, expanded: ExpandedQuery) -> list[str]:
        queries: list[str] = []
        if expanded.english_topic:
            queries.append(expanded.english_topic)
        top = expanded.english_keywords[:8]
        for size in (2, 3):
            for combo in combinations(top, size):
                query = " ".join(combo)
                if len(query) >= 8:
                    queries.append(query)
                if len(queries) >= 12:
                    break
            if len(queries) >= 12:
                break
        return self._dedupe(queries)

    def _build_chinese_queries(self, expanded: ExpandedQuery) -> list[str]:
        queries: list[str] = []
        if expanded.chinese_topic:
            queries.append(expanded.chinese_topic)
        top = expanded.chinese_keywords[:8]
        for size in (2, 3):
            for combo in combinations(top, size):
                query = " ".join(combo)
                if len(query) >= 2:
                    queries.append(query)
                if len(queries) >= 10:
                    break
            if len(queries) >= 10:
                break
        return self._dedupe(queries)

    def _build_boolean_queries(self, expanded: ExpandedQuery) -> list[str]:
        queries: list[str] = []
        en = [k for k in expanded.english_keywords if len(k) > 2]
        if en:
            first_group = en[:2] or ["integrated assessment model"]
            second_group = [k for k in en if "china" in k.lower()][:2] or ["China"]
            third_group = [
                k for k in en
                if "carbon" in k.lower() or "neutrality" in k.lower() or "net zero" in k.lower()
            ][:2] or ["carbon neutrality", "net zero"]
            query = (
                f"(\"{first_group[0]}\""
                + (f" OR \"{first_group[1]}\"" if len(first_group) > 1 else "")
                + ") AND ("
                + " OR ".join(f"\"{x}\"" for x in second_group)
                + ") AND ("
                + " OR ".join(f"\"{x}\"" for x in third_group)
                + ")"
            )
            queries.append(query)

        zh = expanded.chinese_keywords
        if len(zh) >= 3:
            queries.append(f"({zh[0]} OR {zh[1]}) AND ({zh[2]} OR 中国)")

        return self._dedupe(queries)[:8]

    def _build_phrase_queries(self, expanded: ExpandedQuery) -> list[str]:
        phrase_queries: list[str] = [expanded.original_topic]
        if expanded.english_topic and expanded.english_topic != expanded.original_topic:
            phrase_queries.append(expanded.english_topic)
        if expanded.chinese_topic and expanded.chinese_topic != expanded.original_topic:
            phrase_queries.append(expanded.chinese_topic)
        phrase_queries.extend([f"\"{q}\"" for q in self._build_english_queries(expanded)[:4]])
        phrase_queries.extend([f"\"{q}\"" for q in self._build_chinese_queries(expanded)[:4]])
        return self._dedupe(phrase_queries)[:16]

    def _build_loose_queries(self, expanded: ExpandedQuery) -> list[str]:
        queries: list[str] = [expanded.original_topic]
        queries.extend(self._build_english_queries(expanded))
        queries.extend(self._build_chinese_queries(expanded))
        queries.extend(expanded.synonyms[:5])
        queries.extend(expanded.related_terms[:8])
        return self._dedupe([q for q in queries if q and len(q.strip()) > 1])[:30]

    def _extract_english_keywords(self, text: str | None) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-\+_\.]*", text)
        filtered = [token for token in tokens if len(token) > 2 or token.lower() in {"gcam", "iam"}]
        phrases = re.findall(
            r"(carbon neutrality|net zero|power sector|electricity investment|carbon pricing|integrated assessment model|energy system model)",
            text,
            flags=re.IGNORECASE,
        )
        return self._dedupe([*phrases, *filtered])

    def _extract_chinese_keywords(self, text: str | None) -> list[str]:
        if not text:
            return []
        matched_terms = [term for term in QueryTranslator.ZH_TO_EN if term in text]
        compact = re.sub(r"[，。；、,;/\-\(\)（）]", " ", text)
        chunks = [chunk.strip() for chunk in compact.split() if chunk.strip()]
        zh_chunks = [chunk for chunk in chunks if re.search(r"[\u4e00-\u9fff]", chunk)]
        return self._dedupe([*matched_terms, *zh_chunks])

    def _build_synonyms(self, english_keywords: list[str], chinese_keywords: list[str]) -> list[str]:
        synonyms = []
        if any("carbon neutrality" in k.lower() for k in english_keywords):
            synonyms.extend(["net zero", "decarbonization"])
        if any("carbon price" in k.lower() or "carbon pricing" in k.lower() for k in english_keywords):
            synonyms.extend(["carbon market", "emissions trading"])
        if "碳中和" in chinese_keywords:
            synonyms.extend(["净零", "低碳转型"])
        if "碳价格" in chinese_keywords or "碳定价" in chinese_keywords:
            synonyms.extend(["碳市场", "碳交易"])
        return self._dedupe(synonyms)

    def _build_related_terms(self, english_keywords: list[str], chinese_keywords: list[str]) -> list[str]:
        related = list(self.DEFAULT_RELATED_TERMS)
        if any("china" in k.lower() for k in english_keywords) or "中国" in chinese_keywords:
            related.extend(["Chinese energy transition", "中国 能源转型"])
        if any("gcam" in k.lower() for k in english_keywords):
            related.extend(["IAM-based scenario analysis", "integrated assessment scenario"])
        return self._dedupe(related)

    def _llm_available(self) -> bool:
        return bool(self.llm_scorer and getattr(self.llm_scorer, "api_key", None))

    def _llm_expand(self, topic: str) -> dict[str, Any] | None:
        if not self._llm_available():
            return None
        prompt = (
            "You are a bilingual academic query expansion assistant.\n"
            "Given a research topic, return strict JSON with keys:\n"
            "english_topic, chinese_topic, english_keywords, chinese_keywords, synonyms, related_terms.\n"
            "Each keyword list should have up to 10 terms.\n"
            f"topic: {topic}"
        )
        try:
            response = self.llm_scorer._call_api(prompt)  # noqa: SLF001
            if not response:
                return None
            content = (
                (response.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not content:
                return None
            text = content.strip().replace("```json", "").replace("```", "")
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            return data
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM query expansion failed, fallback to rule-based: %s", exc)
            return None

    def _merge_llm_update(self, expanded: ExpandedQuery, llm_data: dict[str, Any]) -> ExpandedQuery:
        model = expanded.model_copy(deep=True)
        model.english_topic = llm_data.get("english_topic") or model.english_topic
        model.chinese_topic = llm_data.get("chinese_topic") or model.chinese_topic
        model.english_keywords = self._dedupe(model.english_keywords + self._to_list(llm_data.get("english_keywords")))
        model.chinese_keywords = self._dedupe(model.chinese_keywords + self._to_list(llm_data.get("chinese_keywords")))
        model.synonyms = self._dedupe(model.synonyms + self._to_list(llm_data.get("synonyms")))
        model.related_terms = self._dedupe(model.related_terms + self._to_list(llm_data.get("related_terms")))
        return model

    def _to_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if not value:
                continue
            norm = re.sub(r"\s+", " ", str(value).strip())
            key = norm.lower()
            if key not in seen:
                seen.add(key)
                result.append(norm)
        return result

