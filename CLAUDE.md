# Transm — Agent Instructions

## Repository Workflow

This repo uses a **fork-based contribution model** for security isolation.

- **Upstream** (TeeYum): `https://github.com/TeeYum/transm` — the canonical repo, owned by TeeYum
- **Fork** (chryst-monode): `https://github.com/chryst-monode/transm` — the working fork on the dev machine

### How commits flow

1. All work happens on the chryst-monode fork (`origin`)
2. Push feature branches to `origin`, never directly to `upstream`
3. Open PRs from `chryst-monode/transm` → `TeeYum/transm`
4. TeeYum reviews and merges — no agent can merge to upstream

### Why this exists

The dev machine (Mac Mini) runs AI agents (NanoClaw, Claude Code, experiments). Only chryst-monode credentials exist on this machine. The fork model ensures agents can propose changes but never unilaterally modify TeeYum's repos.

### Commit identity

- **Author**: chryst-monode (global git identity on this machine)
- **Co-authors**: TeeYum + Claude (auto-added by global `prepare-commit-msg` hook via `.co-authors`)
- Do NOT override the local git user.name/user.email — the global identity and hook handle everything

### Syncing with upstream

```sh
git fetch upstream
git merge upstream/main
```

## Agent Coordination

Before starting work, **read `agent-updates.md`** at the repo root. After completing work, **update it** with what you changed. See that file for format.

## Project Details

- **License**: GPL-3.0 (required by pedalboard and matchering dependencies)
- **Language**: Python (planned)
- **Status**: Pre-alpha / Research Phase
