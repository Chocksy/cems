"""Knowledge graph store using Kuzu for relationship tracking.

The graph store complements vector search by tracking relationships between
memories. This enables queries like:
- "What topics are related to Python?"
- "What preferences affect my coding style?"
- "Show me the relationship graph for project X"

Schema:
    Nodes:
        - Memory: A single memory item
        - Entity: Extracted entity (person, tool, concept)
        - Category: Memory category
        - Tag: Memory tag

    Edges:
        - RELATES_TO: Memory relates to another memory (semantic)
        - MENTIONS: Memory mentions an entity
        - BELONGS_TO: Memory belongs to a category
        - TAGGED_WITH: Memory tagged with a tag
        - ENTITY_LINK: Entity links to another entity
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    label: str  # "Memory", "Entity", "Category", "Tag"
    properties: dict[str, Any]


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    label: str  # "RELATES_TO", "MENTIONS", etc.
    properties: dict[str, Any]


@dataclass
class GraphPath:
    """A path through the graph (sequence of nodes and edges)."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class KuzuGraphStore:
    """Knowledge graph store using Kuzu embedded database.

    Provides graph-based relationship tracking for memories, enabling
    complex queries about how information relates to each other.

    Example:
        graph = KuzuGraphStore("/path/to/graph")
        graph.add_memory_node("mem123", "User prefers Python")
        graph.add_entity("python", "programming_language")
        graph.add_edge("mem123", "python", "MENTIONS")
    """

    def __init__(self, db_path: Path | str):
        """Initialize the Kuzu graph store.

        Args:
            db_path: Path to the Kuzu database directory
        """
        import kuzu

        self.db_path = Path(db_path)
        # Only create parent directories - Kuzu will create the database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = kuzu.Database(str(self.db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the graph schema if not exists."""
        # Create node tables
        node_tables = [
            """
            CREATE NODE TABLE IF NOT EXISTS Memory (
                id STRING PRIMARY KEY,
                content STRING,
                scope STRING,
                user_id STRING,
                created_at STRING,
                updated_at STRING
            )
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Entity (
                id STRING PRIMARY KEY,
                name STRING,
                type STRING,
                created_at STRING
            )
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Category (
                id STRING PRIMARY KEY,
                name STRING,
                scope STRING
            )
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Tag (
                id STRING PRIMARY KEY,
                name STRING
            )
            """,
        ]

        # Create relationship tables
        edge_tables = [
            """
            CREATE REL TABLE IF NOT EXISTS RELATES_TO (
                FROM Memory TO Memory,
                similarity DOUBLE,
                created_at STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS MENTIONS (
                FROM Memory TO Entity,
                confidence DOUBLE,
                created_at STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS BELONGS_TO (
                FROM Memory TO Category,
                created_at STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS TAGGED_WITH (
                FROM Memory TO Tag,
                created_at STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS ENTITY_LINK (
                FROM Entity TO Entity,
                relation_type STRING,
                created_at STRING
            )
            """,
        ]

        for sql in node_tables + edge_tables:
            try:
                self._conn.execute(sql)
            except Exception as e:
                # Ignore errors if tables already exist
                if "already exists" not in str(e).lower():
                    logger.debug(f"Schema init note: {e}")

    def add_memory_node(
        self,
        memory_id: str,
        content: str,
        scope: str = "personal",
        user_id: str = "default",
    ) -> None:
        """Add a memory node to the graph.

        Args:
            memory_id: Unique memory ID
            content: Memory content
            scope: "personal" or "shared"
            user_id: User ID
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.scope = $scope,
                m.user_id = $user_id,
                m.created_at = $created_at,
                m.updated_at = $updated_at
            """,
            {
                "id": memory_id,
                "content": content,
                "scope": scope,
                "user_id": user_id,
                "created_at": now,
                "updated_at": now,
            },
        )

    def add_entity(self, entity_id: str, name: str, entity_type: str) -> None:
        """Add an entity node to the graph.

        Args:
            entity_id: Unique entity ID
            name: Entity name
            entity_type: Type (person, tool, concept, etc.)
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.created_at = $created_at
            """,
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "created_at": now,
            },
        )

    def add_category(self, category_id: str, name: str, scope: str = "personal") -> None:
        """Add a category node to the graph.

        Args:
            category_id: Unique category ID
            name: Category name
            scope: "personal" or "shared"
        """
        self._conn.execute(
            """
            MERGE (c:Category {id: $id})
            SET c.name = $name,
                c.scope = $scope
            """,
            {"id": category_id, "name": name, "scope": scope},
        )

    def add_tag(self, tag_id: str, name: str) -> None:
        """Add a tag node to the graph.

        Args:
            tag_id: Unique tag ID
            name: Tag name
        """
        self._conn.execute(
            """
            MERGE (t:Tag {id: $id})
            SET t.name = $name
            """,
            {"id": tag_id, "name": name},
        )

    def add_memory_relation(
        self,
        source_id: str,
        target_id: str,
        similarity: float = 0.0,
    ) -> None:
        """Add a relation between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            similarity: Similarity score (0-1)
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MATCH (m1:Memory {id: $source_id}), (m2:Memory {id: $target_id})
            MERGE (m1)-[r:RELATES_TO]->(m2)
            SET r.similarity = $similarity,
                r.created_at = $created_at
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "similarity": similarity,
                "created_at": now,
            },
        )

    def add_memory_entity_mention(
        self,
        memory_id: str,
        entity_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Record that a memory mentions an entity.

        Args:
            memory_id: Memory ID
            entity_id: Entity ID
            confidence: Confidence score (0-1)
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MATCH (m:Memory {id: $memory_id}), (e:Entity {id: $entity_id})
            MERGE (m)-[r:MENTIONS]->(e)
            SET r.confidence = $confidence,
                r.created_at = $created_at
            """,
            {
                "memory_id": memory_id,
                "entity_id": entity_id,
                "confidence": confidence,
                "created_at": now,
            },
        )

    def add_memory_to_category(self, memory_id: str, category_id: str) -> None:
        """Add a memory to a category.

        Args:
            memory_id: Memory ID
            category_id: Category ID
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MATCH (m:Memory {id: $memory_id}), (c:Category {id: $category_id})
            MERGE (m)-[r:BELONGS_TO]->(c)
            SET r.created_at = $created_at
            """,
            {
                "memory_id": memory_id,
                "category_id": category_id,
                "created_at": now,
            },
        )

    def add_memory_tag(self, memory_id: str, tag_id: str) -> None:
        """Tag a memory.

        Args:
            memory_id: Memory ID
            tag_id: Tag ID
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MATCH (m:Memory {id: $memory_id}), (t:Tag {id: $tag_id})
            MERGE (m)-[r:TAGGED_WITH]->(t)
            SET r.created_at = $created_at
            """,
            {
                "memory_id": memory_id,
                "tag_id": tag_id,
                "created_at": now,
            },
        )

    def add_entity_link(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
    ) -> None:
        """Link two entities.

        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            relation_type: Type of relation (e.g., "used_for", "part_of")
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            MATCH (e1:Entity {id: $source_id}), (e2:Entity {id: $target_id})
            MERGE (e1)-[r:ENTITY_LINK]->(e2)
            SET r.relation_type = $relation_type,
                r.created_at = $created_at
            """,
            {
                "source_id": source_entity_id,
                "target_id": target_entity_id,
                "relation_type": relation_type,
                "created_at": now,
            },
        )

    def get_related_memories(
        self,
        memory_id: str,
        max_depth: int = 2,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find memories related to a given memory.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum path length to traverse
            limit: Maximum results

        Returns:
            List of related memories with paths
        """
        result = self._conn.execute(
            f"""
            MATCH (m1:Memory {{id: $memory_id}})-[r:RELATES_TO*1..{max_depth}]-(m2:Memory)
            WHERE m1.id <> m2.id
            RETURN m2.id, m2.content, m2.scope, length(r) AS distance
            ORDER BY distance ASC
            LIMIT $limit
            """,
            {"memory_id": memory_id, "limit": limit},
        )

        memories = []
        while result.has_next():
            row = result.get_next()
            memories.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "scope": row[2],
                    "distance": row[3],
                }
            )
        return memories

    def get_memories_by_entity(
        self,
        entity_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find memories that mention a given entity.

        Args:
            entity_id: Entity ID
            limit: Maximum results

        Returns:
            List of memories mentioning the entity
        """
        result = self._conn.execute(
            """
            MATCH (m:Memory)-[r:MENTIONS]->(e:Entity {id: $entity_id})
            RETURN m.id, m.content, m.scope, r.confidence
            ORDER BY r.confidence DESC
            LIMIT $limit
            """,
            {"entity_id": entity_id, "limit": limit},
        )

        memories = []
        while result.has_next():
            row = result.get_next()
            memories.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "scope": row[2],
                    "confidence": row[3],
                }
            )
        return memories

    def get_memories_by_category(
        self,
        category_name: str,
        scope: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find memories in a category.

        Args:
            category_name: Category name
            scope: Optional scope filter
            limit: Maximum results

        Returns:
            List of memories in the category
        """
        if scope:
            query = """
                MATCH (m:Memory)-[:BELONGS_TO]->(c:Category {name: $category_name, scope: $scope})
                RETURN m.id, m.content, m.scope
                LIMIT $limit
            """
            params = {"category_name": category_name, "scope": scope, "limit": limit}
        else:
            query = """
                MATCH (m:Memory)-[:BELONGS_TO]->(c:Category {name: $category_name})
                RETURN m.id, m.content, m.scope
                LIMIT $limit
            """
            params = {"category_name": category_name, "limit": limit}

        result = self._conn.execute(query, params)

        memories = []
        while result.has_next():
            row = result.get_next()
            memories.append({"id": row[0], "content": row[1], "scope": row[2]})
        return memories

    def get_entity_connections(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Find entities connected to a given entity.

        Args:
            entity_id: Entity ID
            max_depth: Maximum path length

        Returns:
            List of connected entities
        """
        result = self._conn.execute(
            f"""
            MATCH (e1:Entity {{id: $entity_id}})-[r:ENTITY_LINK*1..{max_depth}]-(e2:Entity)
            WHERE e1.id <> e2.id
            RETURN e2.id, e2.name, e2.type, length(r) AS distance
            ORDER BY distance
            """,
            {"entity_id": entity_id},
        )

        entities = []
        while result.has_next():
            row = result.get_next()
            entities.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "distance": row[3],
                }
            )
        return entities

    def delete_memory_node(self, memory_id: str) -> None:
        """Delete a memory node and its relationships.

        Args:
            memory_id: Memory ID to delete
        """
        # First delete all relationships
        self._conn.execute(
            """
            MATCH (m:Memory {id: $memory_id})-[r]-()
            DELETE r
            """,
            {"memory_id": memory_id},
        )
        # Then delete the node
        self._conn.execute(
            """
            MATCH (m:Memory {id: $memory_id})
            DELETE m
            """,
            {"memory_id": memory_id},
        )

    def get_graph_stats(self) -> dict[str, int]:
        """Get statistics about the graph.

        Returns:
            Dict with node and edge counts
        """
        stats = {}

        # Count nodes by type
        for label in ["Memory", "Entity", "Category", "Tag"]:
            result = self._conn.execute(f"MATCH (n:{label}) RETURN count(n)")
            if result.has_next():
                stats[f"{label.lower()}_count"] = result.get_next()[0]

        # Count edges by type
        for label in ["RELATES_TO", "MENTIONS", "BELONGS_TO", "TAGGED_WITH", "ENTITY_LINK"]:
            result = self._conn.execute(f"MATCH ()-[r:{label}]->() RETURN count(r)")
            if result.has_next():
                stats[f"{label.lower()}_count"] = result.get_next()[0]

        return stats

    def extract_entities_from_content(self, content: str) -> list[dict[str, str]]:
        """Extract potential entities from memory content.

        Simple pattern-based extraction. For production, consider using
        an LLM or NER model.

        Args:
            content: Memory content text

        Returns:
            List of extracted entities with id, name, type
        """
        entities = []

        # Extract programming languages/tools (simple patterns)
        tools_pattern = r"\b(Python|JavaScript|TypeScript|Rust|Go|Java|Ruby|PHP|Swift|Kotlin|React|Vue|Angular|Node\.js|Django|FastAPI|Flask|Rails|Docker|Kubernetes|Git|PostgreSQL|MongoDB|Redis|AWS|GCP|Azure)\b"
        for match in re.finditer(tools_pattern, content, re.IGNORECASE):
            name = match.group(1)
            entity_id = f"tool:{name.lower()}"
            entities.append(
                {
                    "id": entity_id,
                    "name": name,
                    "type": "tool",
                }
            )

        # Extract concepts (patterns like "prefers X", "likes Y", "uses Z")
        concept_pattern = r"\b(?:prefer|like|use|enjoy|avoid|dislike)s?\s+(\w+(?:\s+\w+)?)"
        for match in re.finditer(concept_pattern, content, re.IGNORECASE):
            name = match.group(1)
            if len(name) > 2:  # Skip very short matches
                entity_id = f"concept:{name.lower().replace(' ', '_')}"
                entities.append(
                    {
                        "id": entity_id,
                        "name": name,
                        "type": "concept",
                    }
                )

        return entities

    def process_memory(
        self,
        memory_id: str,
        content: str,
        scope: str,
        user_id: str,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Process a memory and add it to the graph with relationships.

        This is the main entry point for adding memories with automatic
        entity extraction and relationship creation.

        Args:
            memory_id: Memory ID
            content: Memory content
            scope: "personal" or "shared"
            user_id: User ID
            category: Optional category
            tags: Optional tags
        """
        # Add memory node
        self.add_memory_node(memory_id, content, scope, user_id)

        # Add category if provided
        if category:
            category_id = f"cat:{category}:{scope}"
            self.add_category(category_id, category, scope)
            self.add_memory_to_category(memory_id, category_id)

        # Add tags if provided
        if tags:
            for tag in tags:
                tag_id = f"tag:{tag.lower()}"
                self.add_tag(tag_id, tag)
                self.add_memory_tag(memory_id, tag_id)

        # Extract and add entities
        entities = self.extract_entities_from_content(content)
        for entity in entities:
            self.add_entity(entity["id"], entity["name"], entity["type"])
            self.add_memory_entity_mention(memory_id, entity["id"])

    def close(self) -> None:
        """Close the database connection."""
        # Kuzu handles cleanup automatically
        pass
