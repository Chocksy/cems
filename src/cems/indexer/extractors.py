"""Content extractors for different file types."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedKnowledge:
    """Knowledge extracted from a file."""

    content: str
    file_path: str
    line_range: tuple[int, int] | None = None
    category: str = "general"
    tags: list[str] | None = None
    confidence: float = 1.0


def extract_markdown(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract knowledge from markdown files."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    knowledge = []

    # Extract the whole document as one piece of knowledge
    knowledge.append(
        ExtractedKnowledge(
            content=f"Documentation from {file_path.name}:\n\n{content}",
            file_path=str(file_path),
            category="documentation",
            tags=["docs", file_path.stem.lower()],
        )
    )

    # Also extract individual headers and their content
    sections = re.split(r"^(#+\s+.+)$", content, flags=re.MULTILINE)
    current_header = None
    current_content = []

    for section in sections:
        if section.startswith("#"):
            if current_header and current_content:
                section_text = "\n".join(current_content).strip()
                if len(section_text) > 50:  # Only meaningful sections
                    knowledge.append(
                        ExtractedKnowledge(
                            content=f"{current_header}\n\n{section_text}",
                            file_path=str(file_path),
                            category="documentation",
                            tags=["docs", "section"],
                        )
                    )
            current_header = section.strip()
            current_content = []
        else:
            current_content.append(section)

    return knowledge


def extract_rspec(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract RSpec testing patterns and conventions."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    knowledge = []

    # Extract describe/context blocks with their structure
    describe_pattern = r"(RSpec\.)?describe\s+['\"]?([^'\"]+)['\"]?\s+do"
    describes = re.findall(describe_pattern, content)

    if describes:
        # Summarize what's being tested
        subjects = [d[1] for d in describes]
        knowledge.append(
            ExtractedKnowledge(
                content=f"RSpec tests in {file_path.name} cover: {', '.join(subjects)}",
                file_path=str(file_path),
                category="testing",
                tags=["rspec", "testing"],
            )
        )

    # Extract shared examples
    shared_pattern = r"shared_examples\s+['\"]([^'\"]+)['\"]"
    shared = re.findall(shared_pattern, content)
    if shared:
        knowledge.append(
            ExtractedKnowledge(
                content=f"RSpec shared examples available: {', '.join(shared)}",
                file_path=str(file_path),
                category="testing",
                tags=["rspec", "shared_examples"],
            )
        )

    # Extract let/let! definitions for understanding test setup
    let_pattern = r"let[!]?\s*\(\s*:(\w+)\s*\)"
    lets = list(set(re.findall(let_pattern, content)))
    if lets:
        knowledge.append(
            ExtractedKnowledge(
                content=f"RSpec test fixtures in {file_path.name}: {', '.join(sorted(lets))}",
                file_path=str(file_path),
                category="testing",
                tags=["rspec", "fixtures"],
            )
        )

    # Extract custom matchers from spec_helper/rails_helper
    if "helper" in file_path.name.lower():
        matcher_pattern = r"RSpec::Matchers\.define\s+:(\w+)"
        matchers = re.findall(matcher_pattern, content)
        if matchers:
            knowledge.append(
                ExtractedKnowledge(
                    content=f"Custom RSpec matchers available: {', '.join(matchers)}",
                    file_path=str(file_path),
                    category="testing",
                    tags=["rspec", "matchers", "custom"],
                    confidence=1.0,
                )
            )

        # Extract configuration
        config_block = re.search(r"RSpec\.configure\s+do\s+\|config\|(.*?)end", content, re.DOTALL)
        if config_block:
            knowledge.append(
                ExtractedKnowledge(
                    content=f"RSpec configuration from {file_path.name}:\n\n```ruby\nRSpec.configure do |config|{config_block.group(1)}end\n```",
                    file_path=str(file_path),
                    category="testing",
                    tags=["rspec", "configuration"],
                )
            )

    return knowledge


def extract_adr(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract Architecture Decision Records."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    # ADRs are high-value documents - extract the whole thing
    return [
        ExtractedKnowledge(
            content=f"Architecture Decision Record - {file_path.stem}:\n\n{content}",
            file_path=str(file_path),
            category="architecture",
            tags=["adr", "architecture", "decision"],
            confidence=1.0,
        )
    ]


def extract_ruby_config(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract Ruby/Rails configuration conventions."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    knowledge = []

    if ".rubocop" in file_path.name:
        # Extract rubocop configuration
        knowledge.append(
            ExtractedKnowledge(
                content=f"RuboCop configuration for code style:\n\n```yaml\n{content}\n```",
                file_path=str(file_path),
                category="convention",
                tags=["rubocop", "ruby", "style"],
            )
        )
    elif file_path.suffix == ".rb":
        # Extract initializer purpose from comments
        first_comment = re.match(r"^#[^\n]+", content)
        if first_comment:
            knowledge.append(
                ExtractedKnowledge(
                    content=f"Ruby initializer {file_path.name}: {first_comment.group()}",
                    file_path=str(file_path),
                    category="convention",
                    tags=["ruby", "rails", "initializer"],
                )
            )

    return knowledge


def extract_python_config(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract Python configuration conventions."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    knowledge = []

    if file_path.name == "pyproject.toml":
        knowledge.append(
            ExtractedKnowledge(
                content=f"Python project configuration:\n\n```toml\n{content}\n```",
                file_path=str(file_path),
                category="convention",
                tags=["python", "pyproject", "configuration"],
            )
        )
    elif "conftest" in file_path.name:
        # Extract pytest fixtures
        fixture_pattern = r"@pytest\.fixture[^\n]*\ndef\s+(\w+)"
        fixtures = re.findall(fixture_pattern, content)
        if fixtures:
            knowledge.append(
                ExtractedKnowledge(
                    content=f"Pytest fixtures available: {', '.join(fixtures)}",
                    file_path=str(file_path),
                    category="testing",
                    tags=["pytest", "fixtures"],
                )
            )

    return knowledge


def extract_js_config(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract JavaScript/TypeScript configuration."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    knowledge = []

    if "eslint" in file_path.name.lower():
        knowledge.append(
            ExtractedKnowledge(
                content=f"ESLint configuration for code style:\n\n```json\n{content}\n```",
                file_path=str(file_path),
                category="convention",
                tags=["eslint", "javascript", "typescript", "style"],
            )
        )
    elif "tsconfig" in file_path.name.lower():
        knowledge.append(
            ExtractedKnowledge(
                content=f"TypeScript configuration:\n\n```json\n{content}\n```",
                file_path=str(file_path),
                category="convention",
                tags=["typescript", "configuration"],
            )
        )

    return knowledge


def extract_ci_config(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract CI/CD configuration."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    return [
        ExtractedKnowledge(
            content=f"CI/CD configuration from {file_path.name}:\n\n```yaml\n{content}\n```",
            file_path=str(file_path),
            category="convention",
            tags=["ci", "cd", "automation", file_path.parent.name],
        )
    ]


def extract_schema(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract database schema information."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    knowledge = []

    if file_path.name == "schema.rb":
        # Extract table definitions
        table_pattern = r'create_table\s+"(\w+)"'
        tables = re.findall(table_pattern, content)
        if tables:
            knowledge.append(
                ExtractedKnowledge(
                    content=f"Database tables: {', '.join(sorted(tables))}",
                    file_path=str(file_path),
                    category="architecture",
                    tags=["database", "schema", "rails"],
                )
            )

    # Full schema is valuable
    knowledge.append(
        ExtractedKnowledge(
            content=f"Database schema:\n\n```\n{content[:10000]}\n```",  # Limit size
            file_path=str(file_path),
            category="architecture",
            tags=["database", "schema"],
        )
    )

    return knowledge


def extract_api_spec(file_path: Path) -> list[ExtractedKnowledge]:
    """Extract API specification."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    return [
        ExtractedKnowledge(
            content=f"API specification from {file_path.name}:\n\n```yaml\n{content[:15000]}\n```",
            file_path=str(file_path),
            category="documentation",
            tags=["api", "openapi", "swagger"],
        )
    ]


# Extractor registry
EXTRACTORS = {
    "markdown": extract_markdown,
    "rspec": extract_rspec,
    "adr": extract_adr,
    "ruby_config": extract_ruby_config,
    "python_config": extract_python_config,
    "js_config": extract_js_config,
    "ci_config": extract_ci_config,
    "schema": extract_schema,
    "api_spec": extract_api_spec,
}


def extract_knowledge(file_path: Path, extract_type: str) -> list[ExtractedKnowledge]:
    """Extract knowledge from a file using the appropriate extractor."""
    extractor = EXTRACTORS.get(extract_type)
    if extractor:
        try:
            return extractor(file_path)
        except Exception as e:
            print(f"Warning: Failed to extract from {file_path}: {e}")
            return []
    return []
