from typing import List, Tuple

import sqlalchemy
from oa_configurator import StackConfig, Resolver
from omop_llm import build_model_backend_from_resolved

from ohdsi_question_structurer.backends.postgresql.pg_comparator_selector_backend import \
    PostgresComparatorSelectorBackend


class ComparatorSelectorAdapter:
    def __init__(self, config: StackConfig):
        # Unclear how I use OhdsiQuestionStructurerConfig, so some hacking instead:
        resolver = Resolver(config)
        tool = config.tools.get("ohdsi_question_structurer")
        database = resolver.get_database(tool["comparator_selector_db"])
        connection = resolver.get_connection(database.connection)
        url = sqlalchemy.URL.create(
            drivername=connection.dialect,
            username=connection.user,
            password=connection.password,
            host=connection.host,
            port=connection.port,
            database=connection.database_name,
        )

        model = resolver.resolve_model(tool['embedding_model_name'])
        model_backend = build_model_backend_from_resolved(model)

        engine = sqlalchemy.create_engine(url)
        schema = database.schema_name
        # I'm sure there are clever factory patterns I should use, but for now:
        self.comparator_selector_backend = PostgresComparatorSelectorBackend(engine, schema, model_backend)

    def find_target(self, name: str) -> List[Tuple[int, str]]:
        """
        Find the target comparator for a given name.

        :param name: Name of the target comparator.
        :return List of tuples containing the target ID and name.
        """
        return self.comparator_selector_backend.find_target(name)

    def recommend_comparators(self, target_id: int) -> List[Tuple[float, str]]:
        """
        Recommend comparators for a given target.

        :param target_id: ID of the target comparator.
        :return List of tuples containing the similarity score and comparator name.
        """
        return self.comparator_selector_backend.recommend_comparators(target_id)

    def insert_data(self, cohort_table_path: str, similarity_table_path: str) -> None:
        """
        Insert data into the database.

        :param cohort_table_path: Path to the cohort table. Can be a CSV or gzipped CSV file.
        :param similarity_table_path: Path to the similarity table. Can be a CSV or gzipped CSV file.
        """
        self.comparator_selector_backend.insert_data(cohort_table_path, similarity_table_path)

    def create_embedding_vectors(self) -> None:
        """Create embedding vectors for the comparators using the provided model."""
        self.comparator_selector_backend.create_embedding_vectors()
