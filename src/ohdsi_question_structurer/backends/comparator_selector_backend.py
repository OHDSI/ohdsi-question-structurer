from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from sqlalchemy import Engine
from omop_llm import ModelBackend
import gzip
import pandas as pd

class ComparatorSelector(ABC):

    def __init__(self, engine: Engine, model: ModelBackend) -> None:
        self.engine = engine
        self.model = model

    @abstractmethod
    def find_target(self, name: str) -> List[Tuple[int, str]]:
        """
        Find the target comparator for a given name.

        :param name: Name of the target comparator.
        :return List of tuples containing the target ID and name.
        """
        pass

    @abstractmethod
    def recommend_comparators(self, target_id: int) -> List[Tuple[float, str]]:
        """
        Recommend comparators for a given target.

        :param target_id: ID of the target comparator.
        :return List of tuples containing the similarity score and comparator name.
        """
        pass

    @abstractmethod
    def insert_data(self, cohort_table_path: str, similarity_table_path: str) -> None:
        """
        Insert data into the database.

        :param cohort_table_path: Path to the cohort table. Can be a CSV or gzipped CSV file.
        :param similarity_table_path: Path to the similarity table. Can be a CSV or gzipped CSV file.
        """
        pass

    @abstractmethod
    def create_embedding_vectors(self) -> None:
        """Create embedding vectors for the comparators using the provided model."""
        pass


class PostgresComparatorSelector(ComparatorSelector):
    """PostgreSQL implementation of the ComparatorSelector."""

    def find_target(self, name: str) -> List[Tuple[int, str]]:
        # Implement PostgreSQL-specific logic to find the target comparator
        pass

    def recommend_comparators(self, target_id: int) -> List[Tuple[float, str]]:
        # Implement PostgreSQL-specific logic to recommend comparators
        pass

    def insert_data(self, cohort_table_path: str, similarity_table_path: str) -> None:
        connection = self.engine.connect()
        # Load data into the database using pandas
        with connection.begin():
            # Load cohort data
            if cohort_table_path.endswith('.gz'):
                with gzip.open(cohort_table_path, 'rt') as f:
                    cohort_df = pd.read_csv(f)
            else:
                cohort_df = pd.read_csv(cohort_table_path)
            cohort_df.to_sql('cohort_table', con=connection, if_exists='replace', index=False)

            # Load similarity data
            if similarity_table_path.endswith('.gz'):
                with gzip.open(similarity_table_path, 'rt') as f:
                    similarity_df = pd.read_csv(f)
            else:
                similarity_df = pd.read_csv(similarity_table_path)
            similarity_df.to_sql('similarity_table', con=connection, if_exists='replace', index=False)
        connection.close()

    def create_embedding_vectors(self) -> None:
        connection = self.engine.connect()
        # Create vector table given the embedding vector size:
        