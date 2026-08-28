from __future__ import annotations

import json
from typing import List, Tuple
import logging

import gzip

import numpy as np
import pandas as pd
import requests
from omop_emb.interface import (
    EmbeddingRole,
    EmbeddingReaderInterface,
)
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Column, MetaData, Table, bindparam, case, inspect, or_, select, text

from ohdsi_question_structurer.backends.comparator_selector_backend import ComparatorSelectorBackend

logger = logging.getLogger(__name__)


class PostgresComparatorSelectorBackend(ComparatorSelectorBackend):
    """PostgreSQL implementation of the ComparatorSelectorBackend."""

    COHORT_EMBEDDING_INDEX_NAME = "idx_cs_cohort_embedding_vector_hnsw"
    COHORT_DEFINITION_ID_INDEX_NAME = "idx_cs_cohort_cohort_definition_id"
    SIMILARITY_TARGET_INDEX_NAME = "idx_cs_similarity_cohort_definition_id_1"
    SIMILARITY_COMPARATOR_INDEX_NAME = "idx_cs_similarity_cohort_definition_id_2"
    DEFAULT_FIND_TARGET_LIMIT = 10

    @property
    def _schema_prefix(self) -> str:
        """Returns the schema-qualified prefix (e.g. 'myschema.')."""
        return f"{self.schema}."

    def find_target(self, name: str, top_n: int = DEFAULT_FIND_TARGET_LIMIT) -> List[Tuple[int, str]]:
        if not name.strip() or top_n <= 0:
            return []

        if "jnj.com" in self.model._api_base:
            # Workaround to support the JnJ API:
            payload = json.dumps({
                "input": name
            })
            headers = {
                'api-key': self.model._client.client.api_key,
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", self.model._api_base, headers=headers, data=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get embeddings from JnJ API: {response.status_code} {response.text}")
            query_embedding = np.array([np.array(item["embedding"]) for item in json.loads(response.text)["data"]])
        else:
            query_embedding = EmbeddingReaderInterface.generate_embeddings(
                self.model,
                name,
                role=EmbeddingRole.QUERY,
            )
        embedding_dim = int(query_embedding.shape[1])
        halfvec_type = HALFVEC(embedding_dim)

        with self.engine.connect() as connection:
            inspector = inspect(connection)
            column_names = {
                col["name"]
                for col in inspector.get_columns("cs_cohort", schema=self.schema)
            }
            if "embedding_vector" not in column_names:
                raise RuntimeError(
                    "cs_cohort.embedding_vector does not exist. Run create_embedding_vectors() first."
                )

            rows = connection.execute(
                text(
                    f"""
                    SELECT cohort_definition_id, cohort_name
                    FROM {self._schema_prefix}cs_cohort
                    WHERE embedding_vector IS NOT NULL
                    ORDER BY embedding_vector <=> :query_embedding
                    LIMIT :limit
                    """
                ).bindparams(bindparam("query_embedding", type_=halfvec_type)),
                {
                    "query_embedding": query_embedding[0].tolist(),
                    "limit": top_n,
                },
            ).all()

        return [(int(cohort_definition_id), str(cohort_name)) for cohort_definition_id, cohort_name in rows]

    def recommend_comparators(self, target_id: int, top_n: int = 10, min_databases: int = 1) -> List[Tuple[float, str]]:
        if top_n <= 0:
            return []

        metadata = MetaData()

        cs_similarity = Table(
            "cs_similarity",
            metadata,
            Column("cohort_definition_id_1"),
            Column("cohort_definition_id_2"),
            Column("mean_cosine_similarity"),
            Column("database_count"),
            schema=self.schema,
        )
        cs_cohort = Table(
            "cs_cohort",
            metadata,
            Column("cohort_definition_id"),
            Column("cohort_name"),
            schema=self.schema,
        )

        comparator_id = case(
            (
                cs_similarity.c.cohort_definition_id_1 == target_id,
                cs_similarity.c.cohort_definition_id_2,
            ),
            else_=cs_similarity.c.cohort_definition_id_1,
        )

        stmt = (
            select(cs_similarity.c.mean_cosine_similarity, cs_cohort.c.cohort_name)
            .select_from(
                cs_similarity.join(
                    cs_cohort,
                    cs_cohort.c.cohort_definition_id == comparator_id,
                )
            )
            .where(
                or_(
                    cs_similarity.c.cohort_definition_id_1 == target_id,
                    cs_similarity.c.cohort_definition_id_2 == target_id,
                )
            )
            .where(cs_similarity.c.database_count >= min_databases)
            .order_by(cs_similarity.c.mean_cosine_similarity.desc(), cs_cohort.c.cohort_name.asc())
            .limit(top_n)
        )

        with self.engine.connect() as connection:
            rows = connection.execute(stmt).all()

        return [(float(similarity), str(cohort_name)) for similarity, cohort_name in rows]

    def insert_data(self, cohort_table_path: str, similarity_table_path: str) -> None:
        logger.info("Starting data insert. This may take a while...")

        connection = self.engine.connect()
        with connection.begin():
            # Load cohort data
            if cohort_table_path.endswith('.gz'):
                with gzip.open(cohort_table_path, 'rt') as f:
                    cohort_df = pd.read_csv(f)
            else:
                cohort_df = pd.read_csv(cohort_table_path)

            logger.info("Writing cohort data to %s.cs_cohort", self.schema)
            cohort_df.to_sql("cs_cohort", con=connection, schema=self.schema, if_exists="replace", index=False)

            connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.COHORT_DEFINITION_ID_INDEX_NAME}
                    ON {self._schema_prefix}cs_cohort (cohort_definition_id)
                    """
                )
            )

            # Load similarity data
            if similarity_table_path.endswith('.gz'):
                with gzip.open(similarity_table_path, 'rt') as f:
                    similarity_df = pd.read_csv(f)
            else:
                similarity_df = pd.read_csv(similarity_table_path)

            logger.info("Writing similarity data to %s.cs_similarity", self.schema)
            similarity_df.to_sql("cs_similarity", con=connection, schema=self.schema, if_exists="replace", index=False)

            logger.info("Creating indices")
            connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.SIMILARITY_TARGET_INDEX_NAME}
                    ON {self._schema_prefix}cs_similarity (cohort_definition_id_1)
                    """
                )
            )
            connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.SIMILARITY_COMPARATOR_INDEX_NAME}
                    ON {self._schema_prefix}cs_similarity (cohort_definition_id_2)
                    """
                )
            )
        logger.info("Data insert complete.")
        connection.close()

    def create_embedding_vectors(self) -> None:
        logger.info("Creating embedding vectors. This may take a while...")

        with self.engine.begin() as connection:
            extension_exists = connection.execute(
                text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
            if extension_exists is None:
                try:
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                except Exception as exc:
                    with self.engine.connect() as check_connection:
                        extension_exists = check_connection.execute(
                            text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")
                        ).scalar_one_or_none()
                    if extension_exists is None:
                        raise RuntimeError(
                            "pgvector extension is not available and could not be created. "
                            "Ask a database administrator to run `CREATE EXTENSION vector`."
                        ) from exc
                    logger.warning(
                        "Could not run CREATE EXTENSION for pgvector (likely insufficient privileges), "
                        "but extension already exists. Continuing."
                    )

            cohorts = connection.execute(
                text(
                    f"""
                    SELECT cohort_definition_id, cohort_name
                    FROM {self._schema_prefix}cs_cohort
                    WHERE cohort_name IS NOT NULL
                    ORDER BY cohort_definition_id
                    """
                )
            ).mappings().all()

            if not cohorts:
                return
            if "jnj.com" in self.model._api_base:
                # Workaround to support the JnJ API:
                payload = json.dumps({
                    "input": [str(cohort["cohort_name"]) for cohort in cohorts]
                })
                headers = {
                    'api-key': self.model._client.client.api_key,
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", self.model._api_base, headers=headers, data=payload)
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to get embeddings from JnJ API: {response.status_code} {response.text}")
                embeddings = np.array([np.array(item["embedding"]) for item in json.loads(response.text)["data"]])
            else:
                embeddings = EmbeddingReaderInterface.generate_embeddings(
                    self.model,
                    [str(cohort["cohort_name"]) for cohort in cohorts],
                    role=EmbeddingRole.DOCUMENT,
                )
            embedding_dim = int(embeddings.shape[1])
            halfvec_type = HALFVEC(embedding_dim)
            halfvec_spec = halfvec_type.compile(dialect=connection.dialect)

            inspector = inspect(connection)
            column_names = {
                col["name"]
                for col in inspector.get_columns("cs_cohort", schema=self.schema)
            }

            if "embedding_vector" in column_names:
                existing_type = connection.execute(
                    text(
                        """
                        SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
                        FROM pg_catalog.pg_attribute AS a
                        JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
                        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                        WHERE c.relname = 'cs_cohort'
                          AND n.nspname = :schema_name
                          AND a.attname = 'embedding_vector'
                          AND a.attnum > 0
                          AND NOT a.attisdropped
                        """
                    ),
                    {"schema_name": self.schema},
                ).scalar_one_or_none()
                if existing_type is not None and existing_type.lower() != halfvec_spec.lower():
                    connection.execute(
                        text(f"ALTER TABLE {self._schema_prefix}cs_cohort DROP COLUMN embedding_vector")
                    )
                    column_names.remove("embedding_vector")

            if "embedding_vector" not in column_names:
                connection.execute(
                    text(f"ALTER TABLE {self._schema_prefix}cs_cohort ADD COLUMN embedding_vector {halfvec_spec}")
                )

            connection.execute(
                text(f"UPDATE {self._schema_prefix}cs_cohort SET embedding_vector = NULL WHERE cohort_name IS NULL")
            )

            update_stmt = text(
                f"""
                UPDATE {self._schema_prefix}cs_cohort
                SET embedding_vector = :embedding_vector
                WHERE cohort_definition_id = :cohort_definition_id
                """
            ).bindparams(bindparam("embedding_vector", type_=halfvec_type))

            connection.execute(
                update_stmt,
                [
                    {
                        "cohort_definition_id": cohort["cohort_definition_id"],
                        "embedding_vector": embedding.tolist(),
                    }
                    for cohort, embedding in zip(cohorts, embeddings, strict=True)
                ],
            )

            connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.COHORT_EMBEDDING_INDEX_NAME}
                    ON {self._schema_prefix}cs_cohort
                    USING hnsw (embedding_vector halfvec_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """
                )
            )
