#!/usr/bin/env python3
"""
OAuth wrapper for Graphiti MCP Server
Provides OAuth endpoints required by Claude Code
"""

import logging
import os
from datetime import datetime
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Graphiti MCP OAuth Server")

# OAuth configuration
OAUTH_CONFIG = {
    "client_id": os.environ.get('OAUTH_CLIENT_ID', 'graphiti-mcp'),
    "client_secret": os.environ.get('OAUTH_CLIENT_SECRET', 'graphiti-secret-key-change-this-in-production'),
    "issuer": os.environ.get('OAUTH_ISSUER', 'http://localhost:8020'),
    "audience": os.environ.get('OAUTH_AUDIENCE', 'graphiti-mcp'),
}

# Store registered clients (in production, use a database)
registered_clients: dict[str, dict[str, Any]] = {}


@app.get("/.well-known/oauth-authorization-server")
@app.get("/sse/.well-known/oauth-authorization-server")
async def oauth_metadata():
    """Return OAuth 2.1 authorization server metadata"""
    base_url = OAUTH_CONFIG["issuer"]
    return {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "registration_endpoint": f"{base_url}/register",
        "jwks_uri": f"{base_url}/jwks",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "service_documentation": f"{base_url}/docs",
        "ui_locales_supported": ["en-US"],
    }


@app.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata():
    """Return OAuth protected resource metadata"""
    base_url = OAUTH_CONFIG["issuer"]
    return {
        "resource": base_url,
        "oauth_authorization_server": f"{base_url}/.well-known/oauth-authorization-server",
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{base_url}/docs",
        "resource_signing_alg_values_supported": ["RS256"],
    }


@app.post("/register")
async def register_client(request: Request):
    """Register a new OAuth client"""
    try:
        body = await request.json()

        # Generate client credentials
        import secrets
        client_id = f"client_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)

        client_info = {
            "client_id": client_id,
            "client_secret": client_secret,
            "client_id_issued_at": int(datetime.now().timestamp()),
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            **body
        }

        # Store client info
        registered_clients[client_id] = client_info

        logger.info(f"Registered new client: {client_id}")
        return JSONResponse(client_info, status_code=201)
    except Exception as e:
        logger.error(f"Error registering client: {e}")
        return JSONResponse({"error": "invalid_request"}, status_code=400)


# Proxy SSE endpoint to the actual MCP server
@app.get("/sse")
@app.post("/sse")
async def proxy_sse(request: Request):
    """Proxy SSE requests to the MCP server running on a different port"""
    mcp_port = int(os.environ.get('MCP_INTERNAL_PORT', '2401'))
    mcp_url = f"http://localhost:{mcp_port}/sse"

    # Forward the request headers
    headers = dict(request.headers)
    headers.pop('host', None)  # Remove host header

    if request.method == "GET":
        # Handle SSE streaming
        # Stream the response without buffering
        async def generate():
            # Create httpx client inside the generator to ensure it stays alive
            # Configure timeout for SSE connections (no read timeout for streaming)
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", mcp_url, headers=headers, params=request.query_params) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        # Ensure proper SSE headers
        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=sse_headers
        )
    else:
        # Handle POST requests
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            body = await request.body()
            response = await client.post(
                mcp_url,
                headers=headers,
                content=body,
                follow_redirects=True
            )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )


# Proxy messages endpoint to the actual MCP server
@app.post("/messages/")
async def proxy_messages(request: Request):
    """Proxy messages requests to the MCP server running on a different port"""
    mcp_port = int(os.environ.get('MCP_INTERNAL_PORT', '2401'))
    mcp_url = f"http://localhost:{mcp_port}/messages/"

    # Forward the request headers
    headers = dict(request.headers)
    headers.pop('host', None)  # Remove host header

    # Handle POST requests
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        body = await request.body()
        response = await client.post(
            mcp_url,
            headers=headers,
            content=body,
            params=request.query_params,
            follow_redirects=True
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "Graphiti MCP OAuth Server", "version": "1.0.0"}


if __name__ == "__main__":
    port = int(os.environ.get('MCP_SERVER_PORT', '8020'))
    logger.info(f"Starting OAuth wrapper on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
