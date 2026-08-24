import json
from pathlib import Path

from pydantic import ValidationError

from ohdsi_question_structurer.adapters.comparator_selector_adapter import ComparatorSelectorAdapter
from ohdsi_question_structurer.schemas.study_intent import StudyIntent


class OhdsiQuestionStructurerService:
    def __init__(self, comparator_selector_adapter: ComparatorSelectorAdapter):
        self.comparator_selector_adapter = comparator_selector_adapter

    def find_target(self, name: str):
        """
        Find the target cohort for a given name.

        :param name: Name of the target cohort.
        :return List of tuples containing the target ID and name.
        """
        return self.comparator_selector_adapter.find_target(name)


    def recommend_comparators(self, target_id: int):
        """
        Recommend comparators for a given target cohort.

        :param target_id: ID of the target cohort.
        :return List of tuples containing the similarity score and comparator name.
        """
        return self.comparator_selector_adapter.recommend_comparators(target_id)


    def insert_data(self, cohort_table_path: str, similarity_table_path: str):
        """
        Insert data into the database.

        :param cohort_table_path: Path to the cohort table. Can be a CSV or gzipped CSV file.
        :param similarity_table_path: Path to the similarity table. Can be a CSV or gzipped CSV file.
        """
        self.comparator_selector_adapter.insert_data(cohort_table_path, similarity_table_path)


    def create_embedding_vectors(self):
        """
        Create embedding vectors for the comparators using the provided model.
        """
        self.comparator_selector_adapter.create_embedding_vectors()


    def get_question_structuring_prompt(self, user_input: str) -> str:
        """
        Get the prompt for structuring questions.

        :param user_input: The user's initial question(s) or context.
        :return A string representing the prompt for structuring questions.
        """
        schema_dict = StudyIntent.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2)
        prompt_path = Path(__file__).parent / "prompts" / "ohdsi_question_structurer_prompt.txt"
        prompt_template = prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace("{{schema_json}}", schema_json)
        prompt = prompt.replace("{{user_input}}", user_input)
        return prompt


    def render_study_intent_markdown(self, study_intent: str):
        """
        Render structured questions as pretty Markdown.

        :param structured_questions: A JSON string representing the structured questions.
        :return Markdown string representing the structured questions in a pretty format.
        """
        # Placeholder for actual implementation
        # This method would likely involve processing the questions and returning a structured format
        return f"Structured representation of: {study_intent}"
    

    def  validate_study_intent(self, json_content: str):
        """
        Validate the structured questions.

        :param json_content: A JSON string representing the structured questions.
        :return Validation result indicating whether the questions are valid or not.
        """
        try:
            data = json.loads(json_content)
            StudyIntent.model_validate(data)
            return "Success: The JSON perfectly matches the StudyIntent schema."
        except json.JSONDecodeError as e:
            return f"JSON Formatting Error: {str(e)}"
        except ValidationError as e:
            return f"Schema Validation Error:\n{e.json(indent=2)}"
