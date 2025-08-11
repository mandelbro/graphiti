#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any, cast

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP

# Import configuration classes
from src.config import (
    GraphitiConfig,
    GraphitiEmbedderConfig,
    GraphitiLLMConfig,
    MCPConfig,
    Neo4jConfig,
)

# Import model definitions from the models package
from src.models import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    Preference,
    Procedure,
    Requirement,
    StatusResponse,
    SuccessResponse,
)

# Import memory tools from the tools package
from src.tools import memory_tools

# Import utilities from the utils package
from src.utils import format_fact_result

load_dotenv()


DEFAULT_LLM_MODEL = "deepseek-r1:7b"
SMALL_LLM_MODEL = "deepseek-r1:7b"
DEFAULT_EMBEDDER_MODEL = "nomic-embed-text"

# Semaphore limit for concurrent Graphiti operations.
# Decrease this if you're experiencing 429 rate limit errors from your LLM provider.
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", 10))


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

# Initialize Graphiti client
graphiti_client: Graphiti | None = None


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        # Validate Ollama configuration if using Ollama
        if config.llm.use_ollama:
            if (
                not config.llm.ollama_llm_model
                or not config.llm.ollama_llm_model.strip()
            ):
                raise ValueError(
                    "OLLAMA_LLM_MODEL must be set when using Ollama for LLM"
                )
            logger.info(f"Validated Ollama LLM model: {config.llm.ollama_llm_model}")

        if config.embedder.use_ollama:
            if (
                not config.embedder.ollama_embedding_model
                or not config.embedder.ollama_embedding_model.strip()
            ):
                raise ValueError(
                    "OLLAMA_EMBEDDING_MODEL must be set when using Ollama for embeddings"
                )
            logger.info(
                f"Validated Ollama embedding model: {config.embedder.ollama_embedding_model}"
            )

        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError(
                "OPENAI_API_KEY must be set when custom entities are enabled"
            )

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        embedder_client = config.embedder.create_client()

        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=SEMAPHORE_LIMIT,
        )

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info("Destroying graph...")
            assert graphiti_client is not None
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        assert graphiti_client is not None
        await graphiti_client.build_indices_and_constraints()
        logger.info("Graphiti client initialized successfully")

        # Log configuration details for transparency
        if llm_client:
            if config.llm.use_ollama:
                logger.info(f"Using Ollama LLM model: {config.llm.ollama_llm_model}")
            else:
                logger.info(f"Using OpenAI/Azure OpenAI model: {config.llm.model}")
            logger.info(f"Using temperature: {config.llm.temperature}")
        else:
            logger.info("No LLM client configured - entity extraction will be limited")

        if embedder_client:
            if config.embedder.use_ollama:
                logger.info(
                    f"Using Ollama embedding model: {config.embedder.ollama_embedding_model}"
                )
            else:
                logger.info(
                    f"Using OpenAI/Azure OpenAI embedding model: {config.embedder.model}"
                )
        else:
            logger.info(
                "No embedder client configured - embeddings will not be available"
            )

        logger.info(f"Using group_id: {config.group_id}")
        logger.info(
            f"Custom entity extraction: {'enabled' if config.use_custom_entities else 'disabled'}"
        )
        logger.info(f"Using concurrency limit: {SEMAPHORE_LIMIT}")

        # Set globals for memory tools
        memory_tools.set_globals(graphiti_client, config)

    except Exception as e:
        logger.error(f"Failed to initialize Graphiti: {str(e)}")
        raise


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
    entity: str = "",  # cursor seems to break with None
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.group_id]
            if config.group_id
            else []
        )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != "":
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method
        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message="No relevant nodes found", nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            NodeResult(
                uuid=node.uuid,
                name=node.name,
                summary=node.summary if hasattr(node, "summary") else "",
                labels=node.labels if hasattr(node, "labels") else [],
                group_id=node.group_id,
                created_at=node.created_at.isoformat(),
                attributes=node.attributes if hasattr(node, "attributes") else {},
            )
            for node in search_results.nodes
        ]

        return NodeSearchResponse(
            message="Nodes retrieved successfully", nodes=formatted_nodes
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error searching nodes: {error_msg}")
        return ErrorResponse(error=f"Error searching nodes: {error_msg}")


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error="max_facts must be a positive integer")

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.group_id]
            if config.group_id
            else []
        )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message="No relevant facts found", facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message="Facts retrieved successfully", facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error searching facts: {error_msg}")
        return ErrorResponse(error=f"Error searching facts: {error_msg}")


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
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting entity edge: {error_msg}")
        return ErrorResponse(error=f"Error getting entity edge: {error_msg}")


@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            return ErrorResponse(error="Group ID must be a string")

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id],
            last_n=last_n,
            reference_time=datetime.now(UTC),
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f"No episodes found for group {effective_group_id}", episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode="json")
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting episodes: {error_msg}")
        return ErrorResponse(error=f"Error getting episodes: {error_msg}")


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    return await memory_tools.clear_graph()


@mcp.resource("http://graphiti/status")
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return StatusResponse(status="error", message="Graphiti client not initialized")

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status="ok", message="Graphiti MCP server is running and connected to Neo4j"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error checking Neo4j connection: {error_msg}")
        return StatusResponse(
            status="error",
            message=f"Graphiti MCP server is running but Neo4j connection failed: {error_msg}",
        )


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config

    parser = argparse.ArgumentParser(
        description="Run the Graphiti MCP server with optional LLM client"
    )
    parser.add_argument(
        "--group-id",
        help="Namespace for the graph. This is an arbitrary string used to organize related data. "
        "If not provided, a random UUID will be generated.",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="sse",
        help="Transport to use for communication with the client. (default: sse)",
    )
    parser.add_argument(
        "--model",
        help=f"Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--small-model",
        help=f"Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens for LLM responses (default: 8192)",
    )
    parser.add_argument(
        "--destroy-graph", action="store_true", help="Destroy all Graphiti graphs"
    )
    parser.add_argument(
        "--use-custom-entities",
        action="store_true",
        help="Enable entity extraction using the predefined ENTITY_TYPES",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_SERVER_HOST"),
        help="Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_SERVER_PORT", "8020")),
        help="Port to bind the MCP server to (default: MCP_SERVER_PORT environment variable or 8020)",
    )
    # Ollama configuration arguments
    parser.add_argument(
        "--use-ollama",
        type=lambda x: x.lower() == "true",
        help="Use Ollama for LLM and embeddings (default: true)",
    )
    parser.add_argument(
        "--ollama-base-url",
        help="Ollama base URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--ollama-llm-model",
        help=f"Ollama LLM model name (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        help=f"Ollama embedding model name (default: {DEFAULT_EMBEDDER_MODEL})",
    )
    parser.add_argument(
        "--ollama-embedding-dim",
        type=int,
        help="Ollama embedding dimension (default: 768)",
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f"Using provided group_id: {config.group_id}")
    else:
        logger.info(f"Generated random group_id: {config.group_id}")

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info("Entity extraction enabled using predefined ENTITY_TYPES")
    else:
        logger.info("Entity extraction disabled (no custom entities will be used)")

    # Log LLM configuration
    if config.llm.use_ollama:
        logger.info(f"Using Ollama LLM: {config.llm.ollama_llm_model}")
        logger.info(f"Ollama base URL: {config.llm.ollama_base_url}")
        logger.info(f"LLM temperature: {config.llm.temperature}")
        logger.info(f"LLM max tokens: {config.llm.max_tokens}")
    else:
        logger.info(f"Using OpenAI/Azure OpenAI LLM: {config.llm.model}")
        logger.info(f"LLM temperature: {config.llm.temperature}")
        logger.info(f"LLM max tokens: {config.llm.max_tokens}")

    # Log embedder configuration
    if config.embedder.use_ollama:
        logger.info(f"Using Ollama embedder: {config.embedder.ollama_embedding_model}")
        logger.info(f"Embedding dimension: {config.embedder.ollama_embedding_dim}")
    else:
        logger.info(f"Using OpenAI/Azure OpenAI embedder: {config.embedder.model}")

    # Initialize Graphiti
    await initialize_graphiti()

    if args.host:
        logger.info(f"Setting MCP server host to: {args.host}")
        # Set MCP server host from CLI or env
        mcp.settings.host = args.host

    if args.port:
        logger.info(f"Setting MCP server port to: {args.port}")
        # Set MCP server port from CLI or env
        mcp.settings.port = args.port

    # Return MCP configuration
    return MCPConfig.from_cli(args)


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with stdio transport for MCP in the same event loop
    logger.info(f"Starting MCP server with transport: {mcp_config.transport}")
    if mcp_config.transport == "stdio":
        await mcp.run_stdio_async()
    elif mcp_config.transport == "sse":
        logger.info(
            f"Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}"
        )
        await mcp.run_sse_async()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
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
