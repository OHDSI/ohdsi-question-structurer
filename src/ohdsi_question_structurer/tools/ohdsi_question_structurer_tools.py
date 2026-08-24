import asyncio

from mcp.server import FastMCP

from ohdsi_question_structurer.services.ohdsi_question_structurer_service import OhdsiQuestionStructurerService


def register_ohdsi_question_structurer_tools(server: FastMCP,
                                             ohdsi_question_structurer_service: OhdsiQuestionStructurerService) -> None:
    """
    Register the ohdsi_question_structurer tools with the given FastMCP server.

    This is supposed to use the GroundworkersMCPServer class instead, but I can't see how it is used in the codebase.
    For now, this function will register the ohdsi_question_structurer tools with the provided FastMCP server instance.

    :param server: The FastMCP server instance.
    :param ohdsi_question_structurer_service: The OhdsiQuestionStructurerService instance.
    """

    @server.tool("find_target")
    async def find_target(name: str):
        """
        Find the target cohort for a given name.

        :param name: Name of the target cohort.
        :return List of tuples containing the target ID and name.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.find_target, name)


    @server.tool("recommend_comparators")
    async def recommend_comparators(target_id: int):
        """
        Recommend comparators for a given target cohort.

        :param target_id: ID of the target cohort.
        :return List of tuples containing the similarity score and comparator name.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.recommend_comparators, target_id)


    @server.tool("render_study_intent_markdown")
    async def render_study_intent_markdown(study_intent: str):
        """
        Render structured questions as pretty Markdown.

        :param study_intent: A JSON string representing the structured questions.
        :return Markdown string representing the structured questions in a pretty format.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.render_study_intent_markdown, study_intent)


    @server.resource("resource://ohdsi_question_structurer")
    async def ohdsi_question_structurer() -> str:
        """
        Agent instructions for structuring OHDSI study questions.

        :return A string representing the instructions
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.get_question_structuring_prompt)


    @server.tool("validate_study_intent")
    async def validate_study_intent(study_intent: str):
        """
        Validate the structured questions against the StudyIntent schema.

        :param study_intent: A JSON string representing the structured questions.
        :return A boolean indicating whether the structured questions are valid.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.validate_study_intent, study_intent)


    @server.tool("get_structurer_instructions")
    async def get_structurer_instructions() -> str:
        """
        Retrieve the canonical OHDSI question structuring instructions.

        :return The instruction text used to structure OHDSI study questions.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.get_question_structuring_prompt)
