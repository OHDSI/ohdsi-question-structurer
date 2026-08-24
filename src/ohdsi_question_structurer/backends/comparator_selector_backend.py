from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from omop_llm import ModelBackend
from sqlalchemy import Engine

class ComparatorSelectorBackend(ABC):

    def __init__(self, engine: Engine, schema: str, model: ModelBackend) -> None:
        if not schema or not schema.strip():
            raise ValueError("schema must not be empty")
        self.engine = engine
        self.schema = schema
        self.model = model

    @abstractmethod
    def find_target(self, name: str, top_n: int = 10) -> List[Tuple[int, str]]:
        """
        Find the target cohort for a given name. Uses embedding vectors to find the closest matches.

        :param name: Name of the target comparator.
        :param top_n: Maximum number of matching targets to return.
        :return List of tuples containing the target ID and name.
        """
        pass

    @abstractmethod
    def recommend_comparators(self, target_id: int, top_n: int = 10, min_databases: int = 1) -> List[Tuple[float, str]]:
        """
        Recommend comparators for a given target.

        :param target_id: ID of the target comparator.
        :param top_n: Maximum number of recommended comparators to return.
        :param min_databases: Minimum number of supporting databases required.
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
