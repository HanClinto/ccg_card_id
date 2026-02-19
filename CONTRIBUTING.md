# Contributing

## Commit Discipline

We use small, regular commits and push frequently.

### Rules

1. **Feature-level commits required**
   - When a feature lands (even partial), commit it immediately.
   - Do not keep feature work uncommitted for long periods.

2. **Separate feature work from cleanup**
   - Feature behavior changes in one commit.
   - Refactors/formatting/docs-only cleanup in separate commits.

3. **Push regularly**
   - Push after each feature-level commit (or small batch of tightly related commits).

4. **Commit messages**
   - Use imperative style and be specific:
     - `Add per-image eval cache JSONL for hash retrieval`
     - `Move data path config to env-driven project settings`

5. **For long runs**
   - Commit before starting major long-running jobs.
   - If code changes during runs, stop/restart with a committed state when possible.

6. **Generated artifacts**
   - Derived outputs go under data paths (`$CCG_DATA_DIR/...`), not source tree.
   - Source code/config/docs stay in repo.

## Path Configuration

Use `.env` (copy from `.env.example`) to configure:

- `CCG_DATA_ROOT` (default: `~/claw/data`)
- optional `CCG_DATA_DIR` override
