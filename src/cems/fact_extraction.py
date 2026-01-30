"""Fact extraction for CEMS using LLM.

This module provides fact extraction capabilities that were previously
embedded in Mem0. It extracts actionable facts from conversations using
the CEMS LLM client.

The fact extraction prompt is optimized for developer workflows, extracting:
- Specific commands, file paths, URLs
- Concrete preferences with details
- Workflow steps
- Decisions with context
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.llm import OpenRouterClient

logger = logging.getLogger(__name__)

# The custom fact extraction prompt (migrated from memory.py)
FACT_EXTRACTION_PROMPT = """Extract actionable facts from developer conversations.

For each fact, capture:
- SPECIFIC commands, file paths, URLs (not "uses scripts" but "uses ~/.claude/servers/gsc/seo_daily.sh")
- CONCRETE preferences (not "likes Python" but "prefers Python 3.12+ with strict type hints")
- WORKFLOW steps (not "does SEO" but "runs GSC scripts daily, updates GitHub issue #579 weekly")
- DECISIONS with context (not "uses Coolify" but "deploys EpicPxls via Coolify GitHub integration, never CLI")

Rules:
- Be specific: include file paths, commands, URLs, version numbers
- Be actionable: facts should tell an AI what to DO, not just what exists
- Preserve context: "for EpicPxls", "in production", "when debugging"
- Skip vague context markers like "Context: X" - extract the actual workflow instead
- Deduplicate: if a fact is already captured, don't repeat it
- Each fact should be a complete, standalone statement

Return JSON: {"facts": ["specific actionable fact 1", "specific actionable fact 2"]}"""


class FactExtractor:
    """Extracts actionable facts from text using LLM.

    Example:
        extractor = FactExtractor(client)
        facts = extractor.extract("User said they prefer Python for backend work")
        # Returns: ["User prefers Python for backend development"]
    """

    def __init__(
        self,
        client: "OpenRouterClient | None" = None,
        prompt: str = FACT_EXTRACTION_PROMPT,
    ):
        """Initialize the fact extractor.

        Args:
            client: OpenRouter client. If None, creates one using env vars.
            prompt: Custom fact extraction prompt (optional).
        """
        if client is None:
            from cems.llm import get_client
            client = get_client()

        self._client = client
        self._prompt = prompt

    def extract(
        self,
        content: str,
        context: str | None = None,
        max_facts: int = 10,
    ) -> list[str]:
        """Extract facts from content.

        Args:
            content: The content to extract facts from
            context: Optional additional context
            max_facts: Maximum number of facts to extract

        Returns:
            List of extracted fact strings
        """
        if not content or len(content.strip()) < 10:
            return []

        # Build prompt
        user_prompt = f"Content:\n{content}"
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"

        user_prompt += "\n\nExtract the facts as JSON:"

        try:
            response = self._client.complete(
                prompt=user_prompt,
                system=self._prompt,
                max_tokens=500,
                temperature=0.2,
            )

            facts = self._parse_facts(response)
            return facts[:max_facts]

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return []

    def extract_batch(
        self,
        contents: list[str],
        max_facts_per_item: int = 5,
    ) -> list[list[str]]:
        """Extract facts from multiple content items.

        Args:
            contents: List of content strings
            max_facts_per_item: Maximum facts per item

        Returns:
            List of fact lists (one per content item)
        """
        return [
            self.extract(content, max_facts=max_facts_per_item)
            for content in contents
        ]

    def extract_with_dedup(
        self,
        content: str,
        existing_facts: list[str] | None = None,
    ) -> list[str]:
        """Extract facts while avoiding duplicates with existing facts.

        Args:
            content: The content to extract facts from
            existing_facts: List of existing facts to check for duplicates

        Returns:
            List of new (non-duplicate) facts
        """
        if not existing_facts:
            return self.extract(content)

        # Include existing facts in context so LLM can deduplicate
        context = "Existing facts (do not repeat):\n" + "\n".join(
            f"- {f}" for f in existing_facts[:20]  # Limit to avoid prompt overflow
        )

        return self.extract(content, context=context)

    def _parse_facts(self, response: str) -> list[str]:
        """Parse facts from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            List of fact strings
        """
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                response = match.group(1).strip()

        try:
            data = json.loads(response)

            if isinstance(data, dict) and "facts" in data:
                facts = data["facts"]
            elif isinstance(data, list):
                facts = data
            else:
                logger.warning(f"Unexpected response format: {type(data)}")
                return []

            # Validate and clean facts
            valid_facts = []
            for fact in facts:
                if isinstance(fact, str) and len(fact.strip()) > 5:
                    valid_facts.append(fact.strip())

            return valid_facts

        except json.JSONDecodeError:
            # Try to extract facts from plain text
            logger.debug("JSON parsing failed, trying plain text extraction")
            return self._extract_from_text(response)

    def _extract_from_text(self, text: str) -> list[str]:
        """Extract facts from plain text response (fallback).

        Args:
            text: Plain text response

        Returns:
            List of fact strings
        """
        facts = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            # Skip empty lines and headers
            if not line or line.startswith("#"):
                continue
            # Remove bullet points and numbers
            line = re.sub(r"^[-*•]\s*", "", line)
            line = re.sub(r"^\d+[.)]\s*", "", line)
            # Add if it looks like a fact
            if len(line) > 10:
                facts.append(line)

        return facts


# Module-level instance (lazy initialization)
_extractor: FactExtractor | None = None


def get_fact_extractor() -> FactExtractor:
    """Get the shared FactExtractor instance.

    Returns:
        FactExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = FactExtractor()
    return _extractor


def extract_facts(content: str, context: str | None = None) -> list[str]:
    """Convenience function to extract facts from content.

    Args:
        content: Content to extract facts from
        context: Optional additional context

    Returns:
        List of extracted fact strings
    """
    return get_fact_extractor().extract(content, context)
