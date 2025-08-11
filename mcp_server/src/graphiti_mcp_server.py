#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import configuration classes
from src.config import (
    GraphitiConfig,
    GraphitiEmbedderConfig,
    GraphitiLLMConfig,
    Neo4jConfig,
)

# Import initialization functions
from src.initialization import run_mcp_server

# Import model definitions from the models package
from src.models import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeSearchResponse,
    Preference,
    Procedure,
    Requirement,
    StatusResponse,
    SuccessResponse,
)

# Import tools from the tools package
from src.tools import management_tools, memory_tools
from src.tools import search_memory_facts as search_tools_search_memory_facts
from src.tools import search_memory_nodes as search_tools_search_memory_nodes

load_dotenv()


# Server configuration classes
# The configuration system has a hierarchy:
# - GraphitiConfig is the top-level configuration
#   - LLMConfig handles all OpenAI/LLM related settings
#   - EmbedderConfig manages embedding settings
#   - Neo4jConfig manages database connection details
#   - Various other settings like group_id and feature flags
# Configuration values are loaded from:
# 1. Default values in the class definitions
# 2. Environment variables (loaded via load_dotenv())
# 3. Command line arguments (which override environment variables)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to
capture relationships between concepts, entities, and information. The system organizes data as episodes
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic,
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations.
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    "Graphiti Agent Memory",
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Set default port from environment variable if available
default_port = int(os.environ.get("MCP_SERVER_PORT", "8020"))
mcp.settings.port = default_port


# Register memory tools with MCP decorators
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = "text",
    source_description: str = "",
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph."""
    return await memory_tools.add_memory(
        name, episode_body, group_id, source, source_description, uuid
    )


@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = "",
) -> NodeSearchResponse | ErrorResponse:
    return await search_tools_search_memory_nodes(
        query=query,
        group_ids=group_ids,
        max_nodes=max_nodes,
        center_node_uuid=center_node_uuid,
        entity=entity,
    )


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    return await search_tools_search_memory_facts(
        query=query,
        group_ids=group_ids,
        max_facts=max_facts,
        center_node_uuid=center_node_uuid,
    )


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory."""
    return await memory_tools.delete_entity_edge(uuid)


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory."""
    return await memory_tools.delete_episode(uuid)


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID."""
    return await management_tools.get_entity_edge(uuid)


@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group."""
    return await management_tools.get_episodes(group_id, last_n)


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    return await memory_tools.clear_graph()


@mcp.resource("http://graphiti/status")
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    return await management_tools.get_status()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server(mcp))
    except Exception as e:
        logger.error(f"Error initializing Graphiti MCP server: {str(e)}")
        raise


# Export all the symbols that should be available when importing this module
__all__ = [
    # MCP server instance
    "mcp",
    # Tool functions
    "search_memory_nodes",
    "search_memory_facts",
    "add_memory",
    "get_episodes",
    "delete_entity_edge",
    "delete_episode",
    "get_entity_edge",
    "clear_graph",
    # Configuration classes
    "GraphitiConfig",
    "GraphitiLLMConfig",
    "GraphitiEmbedderConfig",
    "Neo4jConfig",
    # Response models
    "ErrorResponse",
    "SuccessResponse",
    "NodeSearchResponse",
    "FactSearchResponse",
    "EpisodeSearchResponse",
    "StatusResponse",
    # Entity types
    "Preference",
    "Procedure",
    "Requirement",
]


if __name__ == "__main__":
    main()
