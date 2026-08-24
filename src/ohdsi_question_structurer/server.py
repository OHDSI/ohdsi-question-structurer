from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tomllib

from mcp.server.fastmcp import FastMCP
from oa_configurator import StackConfig

from ohdsi_question_structurer.adapters.comparator_selector_adapter import ComparatorSelectorAdapter
from ohdsi_question_structurer.tools.ohdsi_question_structurer_tools import register_ohdsi_question_structurer_tools
from ohdsi_question_structurer.services.ohdsi_question_structurer_service import OhdsiQuestionStructurerService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ohdsi-question-stucturer MCP server")
    parser.add_argument(
        "--config-path",
        help="Path to the shared OMOP stack config TOML. Defaults to OA_CONFIG_PATH or ~/.config/omop/config.toml.",
    )
    parser.add_argument(
        "--profile",
        help="Stack profile override for this process only. Defaults to the stack's active profile.",
    )
    parser.add_argument("--describe", action="store_true", help="Print configured tools and exit")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "rest"],
        help="Transport override. Defaults to tools.groundworkers.mcp.transport for MCP runtimes.",
    )
    parser.add_argument("--host", help="Bind host override for HTTP transports.")
    parser.add_argument("--port", type=int, help="Bind port override for HTTP transports.")
    return parser.parse_args(argv)

def load_stack_config_from_path(path: str | Path) -> StackConfig:
    """Load a stack config from an explicit TOML path."""

    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    try:
        data = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Malformed TOML in {resolved_path}: {exc}") from exc

    config = StackConfig.model_validate(data)
    config.bind_loaded_path(resolved_path)
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_stack_config_from_path(args.config_path)
    comparator_selector_adapter = ComparatorSelectorAdapter(config)
    ohdsi_question_structurer_service = OhdsiQuestionStructurerService(comparator_selector_adapter)

    transport = args.transport
    if transport == "rest":
        raise RuntimeError("Rest transport is not yet supported. Use stdio, sse, or streamable-http instead.")

    app = FastMCP(
        "ohds-question-structurer",
        host=args.host,
        port=args.port,
        json_response=transport == "streamable-http",
        stateless_http=True,
    )

    register_ohdsi_question_structurer_tools(app, ohdsi_question_structurer_service)

    app.run(transport=transport)

if __name__ == "__main__":
    main(sys.argv[1:])