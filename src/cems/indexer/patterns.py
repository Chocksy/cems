"""Index patterns for extracting knowledge from repositories."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IndexPattern:
    """Configuration for extracting knowledge from specific file types."""

    name: str
    description: str
    file_patterns: list[str]
    extract_type: str
    pin_category: str
    extract_comments: bool = False
    extract_docstrings: bool = True
    max_file_size_kb: int = 500
    priority: float = 1.5  # Higher than normal memories


def get_default_patterns() -> list[IndexPattern]:
    """Get default index patterns for common frameworks and languages."""
    return [
        # RSpec patterns
        IndexPattern(
            name="rspec_conventions",
            description="Extract RSpec testing patterns and conventions",
            file_patterns=[
                "spec/**/*_spec.rb",
                "spec/spec_helper.rb",
                "spec/rails_helper.rb",
                "spec/support/**/*.rb",
            ],
            extract_type="rspec",
            pin_category="guideline",
            extract_comments=True,
        ),
        # Ruby/Rails patterns
        IndexPattern(
            name="ruby_conventions",
            description="Extract Ruby/Rails coding conventions",
            file_patterns=[
                ".rubocop.yml",
                ".rubocop_todo.yml",
                "config/initializers/**/*.rb",
                "lib/tasks/**/*.rake",
            ],
            extract_type="ruby_config",
            pin_category="convention",
        ),
        # Documentation
        IndexPattern(
            name="readme_docs",
            description="Extract README and documentation",
            file_patterns=[
                "README.md",
                "README.rst",
                "README.txt",
                "docs/**/*.md",
                "documentation/**/*.md",
            ],
            extract_type="markdown",
            pin_category="documentation",
        ),
        # Architecture Decision Records
        IndexPattern(
            name="architecture_decisions",
            description="Extract architecture decision records (ADRs)",
            file_patterns=[
                "docs/adr/**/*.md",
                "docs/architecture/**/*.md",
                "ADR/**/*.md",
                "doc/adr/**/*.md",
                "decisions/**/*.md",
            ],
            extract_type="adr",
            pin_category="architecture",
            priority=2.0,  # Very high priority
        ),
        # Contributing guidelines
        IndexPattern(
            name="contributing_guide",
            description="Extract contribution guidelines",
            file_patterns=[
                "CONTRIBUTING.md",
                ".github/CONTRIBUTING.md",
                "docs/contributing.md",
                "DEVELOPMENT.md",
            ],
            extract_type="markdown",
            pin_category="guideline",
        ),
        # CI/CD configuration
        IndexPattern(
            name="ci_config",
            description="Extract CI/CD configuration patterns",
            file_patterns=[
                ".github/workflows/*.yml",
                ".github/workflows/*.yaml",
                ".gitlab-ci.yml",
                "Jenkinsfile",
                ".circleci/config.yml",
                ".travis.yml",
            ],
            extract_type="ci_config",
            pin_category="convention",
        ),
        # Python patterns
        IndexPattern(
            name="python_conventions",
            description="Extract Python coding conventions",
            file_patterns=[
                "pyproject.toml",
                "setup.cfg",
                ".flake8",
                ".pylintrc",
                "mypy.ini",
                ".pre-commit-config.yaml",
                "conftest.py",
            ],
            extract_type="python_config",
            pin_category="convention",
        ),
        # TypeScript/JavaScript patterns
        IndexPattern(
            name="js_conventions",
            description="Extract JS/TS coding conventions",
            file_patterns=[
                ".eslintrc*",
                ".prettierrc*",
                "tsconfig.json",
                "tsconfig.*.json",
                "jest.config.*",
                "vitest.config.*",
            ],
            extract_type="js_config",
            pin_category="convention",
        ),
        # Database schemas
        IndexPattern(
            name="database_schema",
            description="Extract database schema information",
            file_patterns=[
                "db/schema.rb",
                "db/structure.sql",
                "migrations/**/*.sql",
                "prisma/schema.prisma",
                "drizzle/**/*.ts",
            ],
            extract_type="schema",
            pin_category="architecture",
        ),
        # API documentation
        IndexPattern(
            name="api_docs",
            description="Extract API documentation",
            file_patterns=[
                "openapi.yaml",
                "openapi.json",
                "swagger.yaml",
                "swagger.json",
                "api/**/*.yaml",
            ],
            extract_type="api_spec",
            pin_category="documentation",
        ),
    ]


def match_files(repo_path: Path, pattern: IndexPattern) -> list[Path]:
    """Find files matching a pattern in a repository."""
    import fnmatch

    matched = []
    for file_pattern in pattern.file_patterns:
        # Handle glob patterns
        if "**" in file_pattern:
            matched.extend(repo_path.glob(file_pattern))
        else:
            matched.extend(repo_path.glob(f"**/{file_pattern}"))

    # Filter by file size
    max_size = pattern.max_file_size_kb * 1024
    return [f for f in matched if f.is_file() and f.stat().st_size <= max_size]
