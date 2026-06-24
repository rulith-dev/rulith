#!/usr/bin/env node
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { JsonlSpaceStore } from '../storage/jsonl-space-store.js'
import { MemorySpaceStore } from '../storage/memory-space-store.js'
import { createRulithCoreMcpServer } from './server.js'

// rulith runs as a stdio MCP server: the agent host spawns this process,
// talks JSON-RPC over stdin/stdout, and kills it when the session ends.
// No port, no daemon, no model access, no network - the board is a pure
// local kernel the agent leans on.
//
// Set RULITH_DB (or legacy RULITH_CORE_DB) to a .jsonl path for
// cross-session persistence; without it, working memory lives and dies
// with this process.
const dbPath = process.env.RULITH_DB ?? process.env.RULITH_CORE_DB
const store = dbPath ? new JsonlSpaceStore(dbPath) : new MemorySpaceStore()

const server = createRulithCoreMcpServer(store)
await server.connect(new StdioServerTransport())
