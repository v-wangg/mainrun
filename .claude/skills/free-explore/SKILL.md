---
name: free-explore
description: Enter read-only exploration mode to freely browse and search the codebase without editing files or triggering permission prompts for reads and searches.
---

# Free Explore Mode

You are now in **read-only exploration mode**. Your job is to thoroughly explore the codebase or vault and report findings — never to implement anything.

## Rules

- **Do NOT edit, write, or create any files.** No Edit, Write, or NotebookEdit tool calls.
- **Do NOT run destructive Bash commands.** No `rm`, `mv`, `git checkout`, `git reset`, or anything that modifies state.
- Explore as much as you need without asking for permission on each read or search action.
- Summarize findings as you go so the user can follow along.

## How to Explore — Agent-First Approach

**Delegate exploration to Explore agents.** Do not read files directly in your main context unless the answer is trivially locatable (e.g. a known file path the user just gave you). For everything else, dispatch Explore agents as subagents.

### Why

- **Context hygiene**: Heavy file reading in subagents keeps your main context clean for synthesis and conversation.
- **Parallelism**: Independent questions should be explored simultaneously by launching multiple Explore agents in a single message.
- **Thoroughness**: Explore agents are specialized for codebase search — they use multiple search strategies and naming conventions automatically.

### Workflow

1. **Understand the user's exploration goal.** What are they trying to learn? Break it into concrete questions.

2. **Dispatch Explore agents in parallel.** For each independent question or area, launch an Explore agent with:
   - A clear, specific prompt describing what to find
   - A thoroughness level: `"quick"` for simple lookups, `"medium"` for moderate exploration, `"very thorough"` for comprehensive analysis
   - Instruction that the task is **research only** — no code changes

   Example — if the user asks "How does the sync system work?", you might launch these in parallel:
   - Agent 1: "Find all files related to sync/replication. Map the directory structure and key files. Thoroughness: very thorough"
   - Agent 2: "Find configuration files, environment variables, and settings related to sync. Thoroughness: medium"
   - Agent 3: "Search for documentation or architecture notes about the sync system. Thoroughness: medium"

3. **Synthesize results.** When agents return, combine their findings into a coherent picture for the user. Identify gaps or follow-up questions.

4. **Iterate if needed.** If the first round reveals new questions, dispatch another round of agents. Each round should build on what you've learned, not repeat it.

### When to read directly vs. dispatch an agent

| Situation | Approach |
|-----------|----------|
| User gave you an exact file path | Read it directly |
| Quick check of a single known file | Read it directly |
| "How does X work?" | Dispatch Explore agent(s) |
| "Find all usages of Y" | Dispatch Explore agent |
| "Map out the structure of Z" | Dispatch Explore agent |
| Multiple independent questions | Dispatch multiple agents in parallel |
| Follow-up on a specific detail from agent results | Read directly or dispatch depending on scope |

### Prompt patterns for Explore agents

Write prompts that are specific and self-contained. The agent has no access to your conversation context.

**Good**: "In the vault at /Users/vwang/Obsidian/Terra, find all Templater templates in Extras/Templates/. For each template, summarize its purpose, what metadata fields it sets, and what folder structure it creates. Thoroughness: very thorough. This is research only — do not modify any files."

**Bad**: "Look at the templates" (too vague, no path, no specifics)

## What you can do

- Read any file (prefer delegating to agents for broad exploration)
- Glob for file patterns (fine for quick targeted lookups)
- Grep for code patterns (fine for quick targeted lookups)
- Run read-only shell commands (`ls`, `git log`, `git diff`, `git status`, `wc -l`, `file`, etc.)
- Launch Explore agents for thorough, parallelized investigation
- Explain code, trace call paths, map architecture, answer questions
- Synthesize findings from multiple agents into clear summaries

## How to use

Just tell me what you want to explore. Examples:

- "How does authentication work?"
- "Find all usages of the `UserService` class"
- "Map out the directory structure of `src/`"
- "What tests exist for the billing module?"
- "How are notes organized in this vault?"
- "Trace the data flow from API request to database"
