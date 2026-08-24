import json
from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from ohdsi_question_structurer.adapters.comparator_selector_adapter import ComparatorSelectorAdapter
from ohdsi_question_structurer.schemas.study_intent import StudyIntent


class OhdsiQuestionStructurerService:
    def __init__(self, comparator_selector_adapter: ComparatorSelectorAdapter):
        self.comparator_selector_adapter = comparator_selector_adapter

    def find_target(self, name: str, top_n: int = 10) -> List[Tuple[int, str]]:
        """
        Find the target cohort for a given name. Uses embedding vectors to find the closest matches.

        :param name: Name of the target cohort.
        :param top_n: Maximum number of matching targets to return.
        :return List of tuples containing the target ID and name.
        """
        return self.comparator_selector_adapter.find_target(name, top_n)


    def recommend_comparators(self, target_id: int, top_n: int = 10, min_databases: int = 1):
        """
        Recommend comparators for a given target cohort.

        :param target_id: ID of the target cohort.
        :param top_n: Maximum number of recommended comparators to return.
        :param min_databases: Minimum number of supporting databases required.
        :return List of tuples containing the similarity score and comparator name.
        """
        return self.comparator_selector_adapter.recommend_comparators(target_id, top_n, min_databases)


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


    def get_question_structuring_prompt(self) -> str:
        """
        Get the prompt for structuring questions.

        :return A string representing the prompt for structuring questions.
        """
        schema_dict = StudyIntent.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2)
        prompt_path = Path(__file__).parent.parent / "prompts" / "ohdsi_question_structurer_prompt.txt"
        prompt_template = prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace("{schema_json}", schema_json)
        return prompt


    def render_study_intent_markdown(self, study_intent: str):
        """
        Render structured questions as pretty Markdown.

        :param structured_questions: A JSON string representing the structured questions.
        :return Markdown string representing the structured questions in a pretty format.
        """
        try:
            data = json.loads(study_intent)
            validated = StudyIntent.model_validate(data)
        except json.JSONDecodeError as e:
            return f"JSON Formatting Error: {str(e)}"
        except ValidationError as e:
            return f"Schema Validation Error:\n{e.json(indent=2)}"

        def _pretty_value(value):
            if value is None:
                return "_Not specified_"
            if isinstance(value, list):
                if not value:
                    return "_None_"
                return ", ".join(f"`{item}`" for item in value)
            return f"`{value}`"

        required_fields_by_template = {
            "patient_characterization": ["target_cohort"],
            "treatment_patterns": ["target_cohort", "treatment_cohorts"],
            "outcome_incidence": ["target_cohort", "outcome_cohort", "time_at_risk"],
            "self_controlled_case_series": ["nesting_cohort", "outcome_cohort", "time_at_risk", "target_cohort"],
            "cohort_method": ["nesting_cohort", "target_cohort", "comparator_cohort", "outcome_cohort", "time_at_risk"],
            "patient_level_prediction": ["target_cohort", "outcome_cohort", "time_at_risk"],
        }

        lines = ["# Study Intent", ""]

        if not validated.analyses:
            lines.append("No analyses provided.")
            return "\n".join(lines)

        lines.append(f"Total analyses: **{len(validated.analyses)}**")
        lines.append("")
        lines.append("## Analysis Overview")
        lines.append("| # | Analytics Type | Template |")
        lines.append("|---|---|---|")
        for idx, analysis in enumerate(validated.analyses, start=1):
            lines.append(f"| {idx} | `{analysis.analytics_type.value}` | `{analysis.template_name.value}` |")
        lines.append("")

        for idx, analysis in enumerate(validated.analyses, start=1):
            params = analysis.parameters.model_dump()
            template_name = analysis.template_name.value
            required_fields = required_fields_by_template.get(template_name, [])

            lines.append(f"## Analysis {idx}: `{template_name}`")
            lines.append(f"Analytics type: `{analysis.analytics_type.value}`")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|---|---|")

            if required_fields:
                for field_name in required_fields:
                    lines.append(f"| `{field_name}` | {_pretty_value(params.get(field_name))} |")
            else:
                lines.append("| _No mapped parameters for this template_ | _Not specified_ |")

            lines.append("")

        return "\n".join(lines).rstrip()


    def  validate_study_intent(self, json_content: str) -> str:
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
