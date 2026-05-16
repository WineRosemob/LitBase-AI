from __future__ import annotations

import re


class QueryTranslator:
    """Lightweight rule-based translator for EN/ZH research topics."""

    ZH_TO_EN = {
        "碳中和": "carbon neutrality",
        "净零": "net zero",
        "电力投资": "electricity investment",
        "电力部门": "power sector",
        "电力行业": "power industry",
        "碳价格": "carbon price",
        "碳定价": "carbon pricing",
        "综合评估模型": "integrated assessment model",
        "能源系统模型": "energy system model",
        "中国": "China",
        "情景分析": "scenario analysis",
        "减排路径": "decarbonization pathway",
    }

    EN_TO_ZH = {
        "carbon neutrality": "碳中和",
        "net zero": "净零",
        "electricity investment": "电力投资",
        "power sector": "电力部门",
        "power industry": "电力行业",
        "carbon price": "碳价格",
        "carbon pricing": "碳定价",
        "integrated assessment model": "综合评估模型",
        "energy system model": "能源系统模型",
        "china": "中国",
        "scenario analysis": "情景分析",
        "decarbonization pathway": "减排路径",
        "gcam": "GCAM",
        "gcam-china": "GCAM-China",
    }

    def detect_language(self, text: str) -> str | None:
        """Detect whether text is mostly Chinese or English."""
        if not text:
            return None
        has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        has_en = bool(re.search(r"[A-Za-z]", text))
        if has_zh and not has_en:
            return "zh"
        if has_en and not has_zh:
            return "en"
        if has_en and has_zh:
            return "mixed"
        return None

    def to_english(self, text: str) -> str:
        """Translate known Chinese phrases to English, fallback to original."""
        result = text
        for zh, en in sorted(self.ZH_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(zh, en)
        return re.sub(r"\s+", " ", result).strip()

    def to_chinese(self, text: str) -> str:
        """Translate known English phrases to Chinese, fallback to original."""
        result = text
        for en, zh in sorted(self.EN_TO_ZH.items(), key=lambda x: len(x[0]), reverse=True):
            result = re.sub(re.escape(en), zh, result, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", result).strip()

