# source-info

Flat landing zone for material handed off from an Obsidian vault named Terra which has a wealth of resources (or, occasionally, pulled directly into the repo) that is referenced while working on this task.

## What lives here

Anything sourced from *somewhere else* — papers, articles, blog posts, tutorial excerpts, video transcripts, or reformulated notes derived from the above. Raw or distilled, both are fine.

Category is a field in the note, not a directory. Keep this flat.

## Handoff convention — lives in the vault

The handoff convention (header format, raw vs. reformulated, citation rules) lives in the active Machine Learning orientation in the Terra vault, not in this repo:

`~/Obsidian/Terra/Eudaimonia/Chamber/2026/04 April/2026-04-23 Orientation - Machine Learning.md`
→ § *Vault ↔ Repo Handoff Convention*

Read that section before creating any note in this directory.

## Why vault, not repo

The vault is the single source of truth for ingested material. This directory exists only so Claude operating in the repo can read without round-tripping to the vault filesystem for every reference. When handing off, cite the vault path — future agents can resolve back if a fresher read is needed.
