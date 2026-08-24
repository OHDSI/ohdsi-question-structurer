import asyncio
import json
from typing import Annotated

from mcp.server import FastMCP
from mcp.types import CallToolResult, ResourceLink, TextContent
from pydantic import AnyUrl, BaseModel

from ohdsi_question_structurer.services.ohdsi_question_structurer_service import OhdsiQuestionStructurerService


class TargetMatch(BaseModel):
    id: int
    name: str


class ComparatorRecommendation(BaseModel):
    similarity: float
    name: str


class TargetMatchesResponse(BaseModel):
    matches: list[TargetMatch]


class ComparatorRecommendationsResponse(BaseModel):
    recommendations: list[ComparatorRecommendation]


def _json_tool_result(structured_content: dict, display_payload: list[dict]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(display_payload, indent=2))],
        structuredContent=structured_content,
    )


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
    async def find_target(name: str, top_n: int = 10) -> Annotated[CallToolResult, TargetMatchesResponse]:
        """
        Find the target cohort for a given name. Uses embedding vectors to find the closest matches.

        :param name: Name of the target cohort.
        :param top_n: Maximum number of matching targets to return.
        :return An array of matches, each containing the target ID and name.
        """
        matches = await asyncio.to_thread(ohdsi_question_structurer_service.find_target, name, top_n)
        response = TargetMatchesResponse(
            matches=[TargetMatch(id=target_id, name=target_name) for target_id, target_name in matches]
        )
        return _json_tool_result(
            structured_content=response.model_dump(mode="json"),
            display_payload=[match.model_dump(mode="json") for match in response.matches],
        )


    @server.tool("recommend_comparators")
    async def recommend_comparators(target_id: int, top_n: int = 10, min_databases: int = 1) -> Annotated[CallToolResult, ComparatorRecommendationsResponse]:
        """
        Recommend comparators for a given target cohort.

        :param target_id: ID of the target cohort.
        :param top_n: Maximum number of recommended comparators to return.
        :param min_databases: Minimum number of supporting databases required.
        :return Structured recommendations containing the similarity score and comparator name.
        """
        recommendations = await asyncio.to_thread(
            ohdsi_question_structurer_service.recommend_comparators,
            target_id,
            top_n,
            min_databases,
        )
        response = ComparatorRecommendationsResponse(
            recommendations=[
                ComparatorRecommendation(similarity=round(similarity_score, 3), name=comparator_name)
                for similarity_score, comparator_name in recommendations
            ]
        )
        return _json_tool_result(
            structured_content=response.model_dump(mode="json"),
            display_payload=[recommendation.model_dump(mode="json") for recommendation in response.recommendations],
        )


    @server.tool("render_study_intent_markdown")
    async def render_study_intent_markdown(study_intent: str) -> str:
        """
        Render structured questions as pretty Markdown.

        :param study_intent: A JSON string representing the structured questions.
        :return Markdown string representing the structured questions in a pretty format.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.render_study_intent_markdown, study_intent)


    @server.tool("validate_study_intent")
    async def validate_study_intent(study_intent: str) -> str:
        """
        Validate the structured questions against the StudyIntent schema.

        :param study_intent: A JSON string representing the structured questions.
        :return A boolean indicating whether the structured questions are valid.
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.validate_study_intent, study_intent)

    @server.resource("resource://ohdsi_question_structurer")
    async def ohdsi_question_structurer() -> str:
        """
        Agent instructions for structuring OHDSI study questions.

        :return A string representing the instructions
        """
        return await asyncio.to_thread(ohdsi_question_structurer_service.get_question_structuring_prompt)


    @server.tool("get_structurer_instructions")
    async def get_structurer_instructions() -> CallToolResult:
        """
        Retrieve the canonical OHDSI question structuring instructions.

        Returns a link to the ``resource://ohdsi_question_structurer`` resource rather
        than the instruction text itself, so clients can resolve it through the normal
        resource-read flow instead of caching a large tool result to disk.

        :return A tool result containing a resource link to the instructions resource.
        """
        return CallToolResult(
            content=[
                ResourceLink(
                    type="resource_link",
                    name="ohdsi_question_structurer",
                    title="OHDSI question structuring instructions",
                    uri=AnyUrl("resource://ohdsi_question_structurer"),
                    description="Canonical instructions for structuring OHDSI study questions.",
                    mimeType="text/plain",
                )
            ]
        )
