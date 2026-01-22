/**
 * Express.js MCP Transport Wrapper for CEMS
 *
 * This wrapper handles MCP Streamable HTTP transport while proxying
 * all tool calls to the Python REST API server.
 *
 * Key features:
 * - Stateless Streamable HTTP transport (no session management)
 * - JSON response mode (no SSE streaming)
 * - Proxies all MCP tool calls to Python REST API endpoints
 * - Auth headers extracted from each request (no session caching)
 */

import express, { Request, Response } from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://cems-server:8765";
const PORT = parseInt(process.env.PORT || "8766", 10);

const app = express();

// Middleware to parse JSON bodies
app.use(express.json());

// Health check endpoint (for Docker/Kubernetes)
app.get("/health", async (_req: Request, res: Response) => {
  try {
    // Optionally check Python API health
    const pythonHealth = await fetch(`${PYTHON_API_URL}/health`).catch(() => null);
    res.json({
      status: "healthy",
      service: "cems-mcp-wrapper",
      python_api: pythonHealth?.ok ? "healthy" : "unknown",
    });
  } catch (error) {
    res.status(503).json({
      status: "unhealthy",
      error: String(error),
    });
  }
});

// Ultra-lightweight ping endpoint for heartbeat checks
app.get("/ping", (_req: Request, res: Response) => {
  res.json({ status: "ok" });
});

// Helper function to create and configure MCP server with all tools/resources
function createMcpServer(authHeaders: { authorization?: string; teamId?: string }) {
  const server = new McpServer({
    name: "cems-mcp-wrapper",
    version: "1.0.0",
  });

  // Helper to get auth headers (always from request, no session caching)
  const getAuthHeaders = () => authHeaders;

  // Register tool handlers that proxy to Python REST API
  server.registerTool(
    "memory_add",
      {
        title: "Add Memory",
        description: "Store a memory in personal or shared namespace. Set infer=false for bulk imports (much faster).",
        inputSchema: {
          content: z.string().describe("What to remember"),
          scope: z.enum(["personal", "shared"]).default("personal").describe("Namespace"),
          category: z.string().default("general").describe("Category for organization"),
          tags: z.array(z.string()).default([]).describe("Optional tags"),
          infer: z.boolean().default(true).describe("Use LLM for fact extraction (true) or store raw (false). Use false for bulk imports."),
        },
      },
    async (args) => {
      const auth = getAuthHeaders();
      const response = await fetch(`${PYTHON_API_URL}/api/memory/add`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(auth.authorization && { Authorization: auth.authorization }),
          ...(auth.teamId && { "x-team-id": auth.teamId }),
        },
        body: JSON.stringify(args),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Python API error: ${error}`);
      }

      const result = await response.json();
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
      };
    }
  );

  server.registerTool(
    "memory_search",
      {
        title: "Search Memories",
        description:
          "Search memories using unified 5-stage retrieval pipeline: query synthesis, vector+graph search, relevance filtering, temporal ranking, and token budgeting. Only returns relevant results (filtered by threshold). Use raw=true for debugging.",
        inputSchema: {
          query: z.string().describe("What to search for"),
          scope: z
            .enum(["personal", "shared", "both"])
            .default("both")
            .describe("Namespace to search"),
          max_results: z.number().default(10).describe("Max results (1-20)"),
          max_tokens: z
            .number()
            .default(2000)
            .describe("Token budget for results"),
          enable_graph: z
            .boolean()
            .default(true)
            .describe("Include graph traversal for related memories"),
          enable_query_synthesis: z
            .boolean()
            .default(true)
            .describe("Use LLM to expand query for better retrieval"),
          raw: z
            .boolean()
            .default(false)
            .describe("Debug mode: bypass filtering to see all results"),
        },
      },
      async (args) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(auth.authorization && { Authorization: auth.authorization }),
            ...(auth.teamId && { "x-team-id": auth.teamId }),
          },
          body: JSON.stringify({
            query: args.query,
            limit: args.max_results,
            scope: args.scope,
            max_tokens: args.max_tokens,
            enable_graph: args.enable_graph,
            enable_query_synthesis: args.enable_query_synthesis,
            raw: args.raw,
          }),
        });

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`Python API error: ${error}`);
        }

        const result = await response.json();
        return {
          content: [{ type: "text", text: JSON.stringify(result) }],
        };
      }
  );

  server.registerTool(
    "memory_forget",
      {
        title: "Forget Memory",
        description: "Delete or archive a memory",
        inputSchema: {
          memory_id: z.string().describe("ID of memory to forget"),
          hard_delete: z.boolean().default(false).describe("Permanently delete if true"),
        },
      },
      async (args) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/forget`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(auth.authorization && { Authorization: auth.authorization }),
          },
          body: JSON.stringify(args),
        });

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`Python API error: ${error}`);
        }

        const result = await response.json();
        return {
          content: [{ type: "text", text: JSON.stringify(result) }],
        };
      }
  );

  server.registerTool(
    "memory_update",
      {
        title: "Update Memory",
        description: "Update an existing memory's content",
        inputSchema: {
          memory_id: z.string().describe("ID of the memory to update"),
          content: z.string().describe("New content for the memory"),
        },
      },
      async (args) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/update`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(auth.authorization && { Authorization: auth.authorization }),
          },
          body: JSON.stringify(args),
        });

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`Python API error: ${error}`);
        }

        const result = await response.json();
        return {
          content: [{ type: "text", text: JSON.stringify(result) }],
        };
      }
  );

  server.registerTool(
    "memory_maintenance",
      {
        title: "Run Maintenance",
        description: "Run memory maintenance jobs",
        inputSchema: {
          job_type: z.enum(["consolidation", "summarization", "reindex", "all"]).default("consolidation").describe("Type of maintenance"),
        },
      },
      async (args) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/maintenance`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(auth.authorization && { Authorization: auth.authorization }),
          },
          body: JSON.stringify(args),
        });

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`Python API error: ${error}`);
        }

        const result = await response.json();
        return {
          content: [{ type: "text", text: JSON.stringify(result) }],
        };
      }
  );

  server.registerTool(
    "session_analyze",
      {
        title: "Analyze Session",
        description: "Analyze session content and extract learnings to remember. Works with any format: raw transcripts, summaries, or notes.",
        inputSchema: {
          transcript: z.string().describe("Session content to analyze - can be a transcript, summary, or notes"),
          session_id: z.string().optional().describe("Optional session identifier"),
          working_dir: z.string().optional().describe("Optional working directory for context"),
        },
      },
      async (args) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/session/analyze`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(auth.authorization && { Authorization: auth.authorization }),
          },
          body: JSON.stringify(args),
        });

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`Python API error: ${error}`);
        }

        const result = await response.json();
        return {
          content: [{ type: "text", text: JSON.stringify(result) }],
        };
      }
  );

  // Register resources
  server.registerResource(
    "memory_status",
      "memory://status",
      {
        title: "Memory System Status",
        description: "Current status of the memory system",
        mimeType: "application/json",
      },
      async (uri) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/status`, {
          method: "GET",
          headers: {
            ...(auth.authorization && { Authorization: auth.authorization }),
          },
        });

        if (!response.ok) {
          throw new Error(`Python API error: ${response.statusText}`);
        }

        const result = await response.json();
        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }
  );

  server.registerResource(
    "memory_personal_summary",
      "memory://personal/summary",
      {
        title: "Personal Memory Summary",
        description: "Summary of personal memories",
        mimeType: "application/json",
      },
      async (uri) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/summary/personal`, {
          method: "GET",
          headers: {
            ...(auth.authorization && { Authorization: auth.authorization }),
          },
        });

        if (!response.ok) {
          throw new Error(`Python API error: ${response.statusText}`);
        }

        const result = await response.json();
        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }
  );

  server.registerResource(
    "memory_shared_summary",
      "memory://shared/summary",
      {
        title: "Shared Memory Summary",
        description: "Summary of shared team memories",
        mimeType: "application/json",
      },
      async (uri) => {
        const auth = getAuthHeaders();
        const response = await fetch(`${PYTHON_API_URL}/api/memory/summary/shared`, {
          method: "GET",
          headers: {
            ...(auth.authorization && { Authorization: auth.authorization }),
            ...(auth.teamId && { "x-team-id": auth.teamId }),
          },
        });

        if (!response.ok) {
          throw new Error(`Python API error: ${response.statusText}`);
        }

        const result = await response.json();
        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }
    );

  return server;
}

// MCP endpoint - handles POST (stateless mode - all requests)
app.post("/mcp", async (req: Request, res: Response) => {
  try {
    // Extract auth headers from request (no session caching)
    const authHeaders = {
      authorization: req.headers.authorization as string | undefined,
      teamId: req.headers["x-team-id"] as string | undefined,
    };

    // Create stateless transport for each request
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined, // Stateless mode - no session tracking
      enableJsonResponse: true, // JSON response, no SSE streaming
    });

    // Create server and register all tools/resources
    const server = createMcpServer(authHeaders);

    // Connect server to transport
    await server.connect(transport);

    // Handle the request
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error("Error handling MCP request:", error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: "Internal server error",
        },
        id: null,
      });
    }
  }
});

// GET /mcp - Return 405 to signal stateless mode (no SSE streaming)
// MCP spec: stateless servers MUST return 405 Method Not Allowed for GET requests
app.get("/mcp", (_req: Request, res: Response) => {
  res.status(405).set('Allow', 'POST').send('Method Not Allowed - This is a stateless MCP server. Use POST requests only.');
});

// DELETE /mcp - Return 405 for stateless mode (no session management)
app.delete("/mcp", (_req: Request, res: Response) => {
  res.status(405).set('Allow', 'POST').send('Method Not Allowed - This is a stateless MCP server. No session management.');
});

// Start server
app.listen(PORT, "0.0.0.0", () => {
  console.log(`CEMS MCP Wrapper listening on port ${PORT}`);
  console.log(`Python API URL: ${PYTHON_API_URL}`);
});
