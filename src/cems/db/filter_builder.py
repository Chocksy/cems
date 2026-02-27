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

    def _add_user_team_or(
        self,
        user_id: str,
        team_id: str,
        condition_template: str,
        col_prefix: str = "",
    ) -> None:
        """Add a compound (user_id OR team_id) condition with manual indexing.

        Args:
            user_id: User ID
            team_id: Team ID
            condition_template: Format string with {p}, {uid_idx}, {tid_idx} placeholders
            col_prefix: Column prefix (e.g., "d.")
        """
        uid_idx, tid_idx = self.add_raw_values(UUID(user_id), UUID(team_id))
        self._conditions.append(
            condition_template.format(p=col_prefix, uid_idx=uid_idx, tid_idx=tid_idx)
        )

    def add_ownership_filter(
        self,
        user_id: str | None,
        team_id: str | None,
        scope: str | Literal["both"],
        col_prefix: str = "",
    ) -> "FilterBuilder":
        """Add ownership filter with OR logic for shared visibility.

        Produces correct visibility rules:
        - scope="personal": user_id = X AND scope = 'personal'
        - scope="shared": (user_id = X OR team_id = Y) AND scope = 'shared'
        - scope="both": (user_id = X OR (team_id = Y AND scope = 'shared'))

        For shared/both scopes, a user can see docs they own OR docs
        belonging to their team.

        Args:
            user_id: User ID
            team_id: Team ID (required for shared/both to enable cross-user visibility)
            scope: Memory scope filter
            col_prefix: Column prefix (e.g., "d." for joined queries)

        Returns:
            Self for chaining
        """
        p = col_prefix

        if scope == "personal" or not team_id:
            # Personal scope or no team: just filter by user_id
            if user_id:
                self.add_param(f"{p}user_id = ${{}}", UUID(user_id))
            if scope != "both":
                self.add_param(f"{p}scope = ${{}}", scope)
        elif scope == "shared":
            # Shared: user sees own docs OR team docs, all with scope='shared'
            if user_id:
                self._add_user_team_or(
                    user_id, team_id,
                    "({p}user_id = ${uid_idx} OR {p}team_id = ${tid_idx})",
                    col_prefix=p,
                )
            else:
                self.add_param(f"{p}team_id = ${{}}", UUID(team_id))
            self.add_param(f"{p}scope = ${{}}", "shared")
        elif scope == "both":
            # Both: user sees all own docs OR shared team docs
            if user_id:
                self._add_user_team_or(
                    user_id, team_id,
                    "({p}user_id = ${uid_idx} OR "
                    "({p}team_id = ${tid_idx} AND {p}scope = 'shared'))",
                    col_prefix=p,
                )
            else:
                self.add_param(f"{p}team_id = ${{}}", UUID(team_id))

        return self

    def add_raw_values(self, *values: Any) -> list[int]:
        """Register extra parameter values and return their placeholder indices.

        Use this for LIMIT/OFFSET or other parameters that aren't part of
        the WHERE clause but share the same query parameter list.

        Returns:
            List of parameter indices (e.g., [5, 6] for $5 and $6)
        """
        indices = []
        for v in values:
            indices.append(self._param_idx)
            self._values.append(v)
            self._param_idx += 1
        return indices

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
