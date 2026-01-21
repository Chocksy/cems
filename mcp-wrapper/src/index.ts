/**
 * Express.js MCP Transport Wrapper for CEMS
 *
 * This wrapper handles MCP transport (SSE) reliably while proxying
 * all tool calls to the Python REST API server.
 *
 * Key features:
 * - SSE transport with proper connection cleanup (Context7 pattern)
 * - Heartbeat every 30 seconds to prevent proxy timeouts
 * - Proxies all MCP tool calls to Python REST API endpoints
 * - Handles authentication via Bearer token passthrough
 */

import express, { Request, Response } from "express";
import { randomUUID } from "node:crypto";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://cems-server:8765";
const PORT = parseInt(process.env.PORT || "8766", 10);

const app = express();

// Middleware to parse JSON bodies
app.use(express.json());

// Store active transports by session ID
const transports: Record<string, StreamableHTTPServerTransport> = {};

// Store active SSE heartbeat intervals
const heartbeatIntervals: Record<string, NodeJS.Timeout> = {};

// SSE heartbeat interval (30 seconds - well under Cloudflare's 100s timeout)
const HEARTBEAT_INTERVAL_MS = 30000;

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

// MCP endpoint - handles POST (initialize and requests)
app.post("/mcp", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;
  let transport: StreamableHTTPServerTransport;

  if (sessionId && transports[sessionId]) {
    // Reuse existing session
    transport = transports[sessionId];
  } else if (!sessionId && isInitializeRequest(req.body)) {
    // New session initialization
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (id) => {
        transports[id] = transport;
        console.log("Session initialized:", id);
      },
      onsessionclosed: (id) => {
        delete transports[id];
        console.log("Session closed:", id);
      },
    });

    transport.onclose = () => {
      if (transport.sessionId) {
        delete transports[transport.sessionId];
      }
    };

    // Create MCP server instance
    const server = new McpServer({
      name: "cems-mcp-wrapper",
      version: "1.0.0",
    });

    // Register tool handlers that proxy to Python REST API
    server.registerTool(
      "memory_add",
      {
        title: "Add Memory",
        description: "Store a memory in personal or shared namespace",
        inputSchema: {
          content: z.string().describe("What to remember"),
          scope: z.enum(["personal", "shared"]).default("personal").describe("Namespace"),
          category: z.string().default("general").describe("Category for organization"),
          tags: z.array(z.string()).default([]).describe("Optional tags"),
        },
      },
      async (args) => {
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/add`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader as string }),
            ...(req.headers["x-team-id"] && { "x-team-id": req.headers["x-team-id"] as string }),
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
        description: "Search memories using unified intelligent search",
        inputSchema: {
          query: z.string().describe("What to search for"),
          scope: z.enum(["personal", "shared", "both"]).default("both").describe("Namespace"),
          max_results: z.number().default(10).describe("Max results (1-20)"),
        },
      },
      async (args) => {
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader as string }),
            ...(req.headers["x-team-id"] && { "x-team-id": req.headers["x-team-id"] as string }),
          },
          body: JSON.stringify({
            query: args.query,
            limit: args.max_results,
            scope: args.scope,
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
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/forget`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader as string }),
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
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/update`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader as string }),
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
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/maintenance`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader as string }),
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
        description: "Analyze a raw session transcript and extract learnings to remember. IMPORTANT: Pass the actual conversation text (USER/ASSISTANT messages), NOT a summary. The system uses an LLM to extract patterns from raw dialogue.",
        inputSchema: {
          transcript: z.string().describe("The RAW session transcript with USER and ASSISTANT messages. Do NOT summarize - pass the actual conversation verbatim. Example format: 'USER: How do I fix X?\\nASSISTANT: You can fix it by...'"),
          session_id: z.string().optional().describe("Optional session identifier"),
          working_dir: z.string().optional().describe("Optional working directory for context"),
        },
      },
      async (args) => {
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/session/analyze`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader as string }),
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
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/status`, {
          method: "GET",
          headers: {
            ...(authHeader && { Authorization: authHeader as string }),
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
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/summary/personal`, {
          method: "GET",
          headers: {
            ...(authHeader && { Authorization: authHeader as string }),
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
        const authHeader = req.headers.authorization;
        const response = await fetch(`${PYTHON_API_URL}/api/memory/summary/shared`, {
          method: "GET",
          headers: {
            ...(authHeader && { Authorization: authHeader as string }),
            ...(req.headers["x-team-id"] && { "x-team-id": req.headers["x-team-id"] as string }),
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

    // Connect server to transport
    await server.connect(transport);
  } else {
    res.status(400).json({
      jsonrpc: "2.0",
      error: { code: -32000, message: "Invalid session" },
      id: null,
    });
    return;
  }

  // Handle the request
  await transport.handleRequest(req, res, req.body);
});

// MCP endpoint - handles GET (SSE stream) with heartbeat to prevent Cloudflare timeout
app.get("/mcp", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string;
  const transport = transports[sessionId];

  if (transport) {
    // Set up SSE heartbeat to prevent Cloudflare's 100s idle timeout
    // Send SSE comment (: ping) every 30 seconds
    const heartbeatId = `${sessionId}-${Date.now()}`;
    
    // Start heartbeat interval
    heartbeatIntervals[heartbeatId] = setInterval(() => {
      try {
        if (!res.writableEnded && !res.destroyed) {
          // SSE comment format - doesn't trigger client event but keeps connection alive
          res.write(": ping\n\n");
        } else {
          // Connection closed, clean up
          clearInterval(heartbeatIntervals[heartbeatId]);
          delete heartbeatIntervals[heartbeatId];
        }
      } catch (err) {
        // Connection likely closed
        clearInterval(heartbeatIntervals[heartbeatId]);
        delete heartbeatIntervals[heartbeatId];
      }
    }, HEARTBEAT_INTERVAL_MS);

    // Clean up heartbeat when response ends
    res.on("close", () => {
      if (heartbeatIntervals[heartbeatId]) {
        clearInterval(heartbeatIntervals[heartbeatId]);
        delete heartbeatIntervals[heartbeatId];
      }
    });

    res.on("finish", () => {
      if (heartbeatIntervals[heartbeatId]) {
        clearInterval(heartbeatIntervals[heartbeatId]);
        delete heartbeatIntervals[heartbeatId];
      }
    });

    await transport.handleRequest(req, res);
  } else {
    res.status(400).send("Invalid session");
  }
});

// MCP endpoint - handles DELETE (close session)
app.delete("/mcp", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string;
  const transport = transports[sessionId];

  if (transport) {
    await transport.handleRequest(req, res);
  } else {
    res.status(400).send("Invalid session");
  }
});

// Start server
app.listen(PORT, "0.0.0.0", () => {
  console.log(`CEMS MCP Wrapper listening on port ${PORT}`);
  console.log(`Python API URL: ${PYTHON_API_URL}`);
});
