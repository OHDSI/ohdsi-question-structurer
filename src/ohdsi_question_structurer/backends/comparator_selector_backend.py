from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from omop_llm import ModelBackend
from sqlalchemy import Engine, bindparam, case, column, inspect, or_, select, table, text

class ComparatorSelectorBackend(ABC):

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
