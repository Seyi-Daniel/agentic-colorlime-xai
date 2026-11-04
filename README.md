# Second archived branch snapshots

These folders preserve the branches retired during the second repository reorganization on November 4, 2026.
Each folder contains the exact original Git tree, including hidden files. The archive commit retains
every distinct original branch tip as a parent, preserving complete histories.

Active branches are `main`, `develop`, `seyi/develop_01` and `zain/develop_01`.
The three development branches start from `seyi/zain/develop_merge` at
[`ba274b5313a3`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/ba274b5313a3bb7f90ccfe1645be0d6ab8c4bab7),
with only the shared README clone instruction updated to the new develop branch.
The [first archive](https://github.com/Seyi-Daniel/agentic-colorlime-xai/tree/archive-snapshots) remains unchanged.

| Snapshot folder | Original branch | Original commit |
| --- | --- | --- |
| [seyi__develop](seyi__develop/) | `seyi/develop` | [`8d89f57f3e44`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/8d89f57f3e44e8bc2ce7a452a01e3382ebbfa510) |
| [seyi__zain__develop_jupytext](seyi__zain__develop_jupytext/) | `seyi/zain/develop_jupytext` | [`202580f680ff`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/202580f680ff8fb4f4478242eabfbd69d29531e0) |
| [seyi__zain__develop_merge](seyi__zain__develop_merge/) | `seyi/zain/develop_merge` | [`ba274b5313a3`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/ba274b5313a3bb7f90ccfe1645be0d6ab8c4bab7) |
| [zain__develop_current](zain__develop_current/) | `zain/develop_current` | [`84454b6fcd82`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/84454b6fcd822a040f7c0443de2f0f953ad1f0c2) |
| [zain__develop_jupytext](zain__develop_jupytext/) | `zain/develop_jupytext` | [`202580f680ff`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/202580f680ff8fb4f4478242eabfbd69d29531e0) |
| [zain__develop_merge](zain__develop_merge/) | `zain/develop_merge` | [`86c4e84f53bf`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/86c4e84f53bf35428d20b0a04f04ae81f8ee2c1a) |

## Using an original version

Snapshot folders are historical references. Their instructions may assume they are at the repository root.
For the exact original checkout, clone the archive and check out a commit from the table:

```bash
git clone --branch archive-snapshots-02 https://github.com/Seyi-Daniel/agentic-colorlime-xai.git agentic-xai-history-02
cd agentic-xai-history-02
git switch --detach <original-commit-sha>
```

The previous-to-current Zain import history remains reachable through the original commits.
See [manifest.json](manifest.json) for full commit and tree identifiers.
