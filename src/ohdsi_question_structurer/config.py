"""Configuration for ohdsi-question-structurer via oa-configurator."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import Field
from sqlalchemy import Engine
from oa_configurator import (
    GenericDatabaseConfig,
    ModelConfig,
    PackageConfigBase,
    RefTo
)

# This code is likely wrong, and currently not used

class OhdsiQuestionStructurerConfig(PackageConfigBase):
    """oa-configurator config class for ohdsi-question-structurer.

    Notes
    -----
    By design, this config is for internal use only and must not be
    imported or resolved by any other package.
    """

    tool_name: ClassVar[str] = "ohdsi_question_structurer"
    extra_logging_namespaces: ClassVar[tuple[str, ...]] = ()

    comparator_selector_db: Annotated[str, RefTo(GenericDatabaseConfig)] = "comparator_selector_db"
    embedding_model_name: Annotated[str, RefTo(ModelConfig)] = Field(
        default="qwen3-embedding",
        description=(
            "Name of a [models.*] entry (see 'omop-config models add') to use for "
            "generating concept embeddings."
        ),
    )



def resolve_ohdsi_question_structurer_engine() -> Engine:
    """Resolve engine via oa-configurator"""
    return OhdsiQuestionStructurerConfig.get_engine(OhdsiQuestionStructurerConfig.get_config().db)


class BackendType(StrEnum):
    """Database backend.

    Members
    -------
    SQLITEVEC
        Default backend. Requires no external database server.
    PGVECTOR
        Optional backend. Requires a PostgreSQL instance with the pgvector
        extension (``pip install omop-emb[pgvector]``).
    """

    POSTGRESQL = "postgresql"
    DUCKDB = "duckdb"



def parse_backend_type(value: str | BackendType) -> BackendType:
    """Parse a string or ``BackendType`` into a ``BackendType``.

    Parameters
    ----------
    value : str | BackendType
        Backend identifier string or enum member.

    Returns
    -------
    BackendType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised backend type.
    """
    if isinstance(value, BackendType):
        return value
    try:
        return BackendType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid backend type {value!r}. Expected one of "
            f"{[member.value for member in BackendType]}."
        ) from exc
