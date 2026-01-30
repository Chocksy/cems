"""Filter builder for dynamic WHERE clause construction."""

from typing import Any, Literal
from uuid import UUID


class FilterBuilder:
    """Build dynamic WHERE clauses with automatic parameter indexing.

    Simplifies the common pattern of building SQL WHERE clauses with
    dynamic conditions and parameter placeholders.

    Example:
        fb = FilterBuilder(start_idx=3)  # $1 and $2 reserved for other params
        fb.add_if(user_id, "user_id = ${}", UUID(user_id))
        fb.add_if(scope != "both", "scope = ${}", scope)
        fb.add("archived = FALSE")  # Always added

        where_clause = fb.build()
        all_values = [embedding, limit] + fb.values
    """

    def __init__(self, start_idx: int = 1):
        """Initialize the filter builder.

        Args:
            start_idx: Starting parameter index (e.g., 3 if $1 and $2 are reserved)
        """
        self._conditions: list[str] = []
        self._values: list[Any] = []
        self._param_idx = start_idx

    @property
    def values(self) -> list[Any]:
        """Get the list of parameter values."""
        return self._values

    @property
    def next_idx(self) -> int:
        """Get the next parameter index."""
        return self._param_idx

    def add(self, condition: str) -> "FilterBuilder":
        """Add a condition without parameters.

        Args:
            condition: SQL condition string (e.g., "archived = FALSE")

        Returns:
            Self for chaining
        """
        self._conditions.append(condition)
        return self

    def add_param(self, condition_template: str, value: Any) -> "FilterBuilder":
        """Add a condition with a single parameter.

        Args:
            condition_template: SQL with {} placeholder for param index
                                (e.g., "user_id = ${}")
            value: Parameter value

        Returns:
            Self for chaining
        """
        condition = condition_template.replace("${}", f"${self._param_idx}")
        self._conditions.append(condition)
        self._values.append(value)
        self._param_idx += 1
        return self

    def add_if(
        self,
        condition_check: Any,
        condition_template: str,
        value: Any,
    ) -> "FilterBuilder":
        """Add a condition only if the check is truthy.

        Args:
            condition_check: Value to check (adds condition if truthy)
            condition_template: SQL with {} placeholder for param index
            value: Parameter value

        Returns:
            Self for chaining
        """
        if condition_check:
            self.add_param(condition_template, value)
        return self

    def add_not_archived(self) -> "FilterBuilder":
        """Add standard conditions for excluding archived/expired memories."""
        self.add("archived = FALSE")
        self.add("(expires_at IS NULL OR expires_at > NOW())")
        return self

    def add_scope_filter(
        self,
        scope: str | Literal["both"],
        user_id: str | None = None,
        team_id: str | None = None,
    ) -> "FilterBuilder":
        """Add scope-based filtering conditions.

        Args:
            scope: Memory scope filter ("personal", "shared", "both")
            user_id: User ID to filter by
            team_id: Team ID to filter by (used for shared scope)

        Returns:
            Self for chaining
        """
        if user_id:
            self.add_param("user_id = ${}", UUID(user_id))

        if team_id and scope in ("shared", "both"):
            self.add_param("team_id = ${}", UUID(team_id))

        if scope != "both":
            self.add_param("scope = ${}", scope)

        return self

    def build(self, default: str = "TRUE") -> str:
        """Build the WHERE clause string.

        Args:
            default: Value to return if no conditions (default "TRUE")

        Returns:
            WHERE clause conditions joined by AND
        """
        if not self._conditions:
            return default
        return " AND ".join(self._conditions)

    def __bool__(self) -> bool:
        """Check if any conditions have been added."""
        return bool(self._conditions)

    def __len__(self) -> int:
        """Get the number of conditions."""
        return len(self._conditions)
