"""DuckDB structured store for clinical code data.

This is the agent's primary knowledge base for EXACT code lookups.
When the user provides a specific code (R65.20, DRG 470, etc.), the
agent queries DuckDB directly — no embedding or vector search needed.

Schema design:
- `codes` table: all code systems unified into one table with a
  code_system discriminator. This simplifies the agent's tool interface
  (one query function, not four).
- `code_hierarchy` table: parent-child relationships for navigation.
- `drg_components` table: which diagnoses/procedures trigger each DRG.
"""

from pathlib import Path

import duckdb

from clinical_code_agent.config import settings
from clinical_code_agent.ingestion.parsers import CodeRecord


class CodeStore:
    """DuckDB-backed store for clinical code structured lookups."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = str(db_path or settings.duckdb_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Lazy connection — creates DB file on first access."""
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(self._db_path)
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS codes (
                code VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                code_system VARCHAR NOT NULL,
                chapter VARCHAR DEFAULT '',
                category VARCHAR DEFAULT '',
                is_header BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (code, code_system)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS code_hierarchy (
                parent_code VARCHAR NOT NULL,
                child_code VARCHAR NOT NULL,
                code_system VARCHAR NOT NULL,
                relationship_type VARCHAR DEFAULT 'parent-child',
                PRIMARY KEY (parent_code, child_code, code_system)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS drg_components (
                drg_code VARCHAR NOT NULL,
                component_code VARCHAR NOT NULL,
                component_system VARCHAR NOT NULL,
                component_role VARCHAR DEFAULT 'principal_dx',
                PRIMARY KEY (drg_code, component_code, component_system)
            )
        """)

    def load_records(self, records: list[CodeRecord]) -> int:
        """Insert parsed code records into the codes table.

        Uses INSERT OR REPLACE to handle re-ingestion cleanly.
        Returns the number of records inserted.
        """
        if not records:
            return 0

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO codes (code, description, code_system, chapter, category, is_header)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (r.code, r.description, r.code_system, r.chapter, r.category, r.is_header)
                for r in records
            ],
        )
        return len(records)

    def build_hierarchy(self) -> int:
        """Derive parent-child relationships from code structure.

        ICD-10-CM hierarchy:
        - Category (3 chars, e.g., R65) is parent of subcategories (R65.1, R65.2)
        - Subcategory (4-5 chars) is parent of extensions (R65.20, R65.21)

        This populates code_hierarchy from the existing codes table.
        """
        # Clear existing hierarchy
        self.conn.execute("DELETE FROM code_hierarchy")

        # ICD-10-CM: 3-char category -> 4+ char children
        self.conn.execute("""
            INSERT INTO code_hierarchy (parent_code, child_code, code_system, relationship_type)
            SELECT
                p.code AS parent_code,
                c.code AS child_code,
                'ICD-10-CM' AS code_system,
                'parent-child' AS relationship_type
            FROM codes p
            JOIN codes c ON c.code LIKE p.code || '.%'
                AND c.code_system = 'ICD-10-CM'
            WHERE p.code_system = 'ICD-10-CM'
                AND p.is_header = TRUE
                AND LENGTH(p.code) = 3
        """)

        # ICD-10-PCS: 2-char section+body system -> 7-char codes
        self.conn.execute("""
            INSERT INTO code_hierarchy (parent_code, child_code, code_system, relationship_type)
            SELECT DISTINCT
                LEFT(c.code, 2) AS parent_code,
                c.code AS child_code,
                'ICD-10-PCS' AS code_system,
                'section-member' AS relationship_type
            FROM codes c
            WHERE c.code_system = 'ICD-10-PCS'
                AND c.is_header = FALSE
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM code_hierarchy").fetchone()
        return count[0] if count else 0

    def lookup_code(self, code: str, code_system: str | None = None) -> list[dict[str, object]]:
        """Exact code lookup. Returns matching records.

        If code_system is None, searches all systems.
        Normalizes input (strips dots, uppercases) for flexible matching.
        """
        # Normalize: remove dots for matching, uppercase
        normalized = code.upper().replace(".", "").replace(" ", "")

        if code_system:
            results = self.conn.execute(
                """
                SELECT code, description, code_system, chapter, category, is_header
                FROM codes
                WHERE REPLACE(UPPER(code), '.', '') = ?
                    AND code_system = ?
                """,
                [normalized, code_system],
            ).fetchall()
        else:
            results = self.conn.execute(
                """
                SELECT code, description, code_system, chapter, category, is_header
                FROM codes
                WHERE REPLACE(UPPER(code), '.', '') = ?
                """,
                [normalized],
            ).fetchall()

        return [
            {
                "code": r[0],
                "description": r[1],
                "code_system": r[2],
                "chapter": r[3],
                "category": r[4],
                "is_header": r[5],
                "score": 1.0,  # Exact match = perfect confidence
            }
            for r in results
        ]

    def get_children(self, parent_code: str, code_system: str | None = None) -> list[dict[str, object]]:
        """Get child codes in the hierarchy."""
        normalized = parent_code.upper().replace(".", "").replace(" ", "")

        query = """
            SELECT c.code, c.description, c.code_system, c.chapter, c.category
            FROM code_hierarchy h
            JOIN codes c ON c.code = h.child_code AND c.code_system = h.code_system
            WHERE REPLACE(UPPER(h.parent_code), '.', '') = ?
        """
        params: list[object] = [normalized]

        if code_system:
            query += " AND h.code_system = ?"
            params.append(code_system)

        query += " ORDER BY c.code LIMIT 50"

        results = self.conn.execute(query, params).fetchall()
        return [
            {
                "code": r[0],
                "description": r[1],
                "code_system": r[2],
                "chapter": r[3],
                "category": r[4],
                "score": 1.0,
            }
            for r in results
        ]

    def search_description(self, query: str, code_system: str | None = None, limit: int = 20) -> list[dict[str, object]]:
        """Simple text search on descriptions (fallback when vector store unavailable).

        Uses DuckDB's ILIKE for substring matching. Not as good as vector
        search but works without embeddings for basic testing.
        """
        sql = """
            SELECT code, description, code_system, chapter, category, is_header
            FROM codes
            WHERE LOWER(description) LIKE ?
                AND is_header = FALSE
        """
        params: list[object] = [f"%{query.lower()}%"]

        if code_system:
            sql += " AND code_system = ?"
            params.append(code_system)

        sql += f" ORDER BY LENGTH(description) ASC LIMIT {limit}"

        results = self.conn.execute(sql, params).fetchall()
        return [
            {
                "code": r[0],
                "description": r[1],
                "code_system": r[2],
                "chapter": r[3],
                "category": r[4],
                "is_header": r[5],
                "score": 0.7,  # Text match is lower confidence than exact
            }
            for r in results
        ]

    @property
    def stats(self) -> dict[str, int]:
        """Return counts of loaded data."""
        total = self.conn.execute("SELECT COUNT(*) FROM codes").fetchone()
        by_system = self.conn.execute(
            "SELECT code_system, COUNT(*) FROM codes GROUP BY code_system"
        ).fetchall()
        hierarchy = self.conn.execute("SELECT COUNT(*) FROM code_hierarchy").fetchone()

        result = {
            "total_codes": total[0] if total else 0,
            "hierarchy_relationships": hierarchy[0] if hierarchy else 0,
        }
        for system, count in by_system:
            result[f"codes_{system.lower().replace('-', '_')}"] = count

        return result

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
