# Archived branch snapshots

The active Jupytext project is on [main](https://github.com/Seyi-Daniel/agentic-colorlime-xai/tree/main).
This branch consolidates the other GitHub branches as they stood on November 4, 2026.

Each folder contains the exact Git tree from its original branch tip, including hidden files.
The archive commit also retains those branch tips as parents, keeping their complete commit history reachable.

| Folder | Original branch | Original commit |
| --- | --- | --- |
| [archive__main-before-notebook-2026-10-13](archive__main-before-notebook-2026-10-13/) | `archive/main-before-notebook-2026-10-13` | [`821bf0ed0523`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/821bf0ed0523ec7feac4847684cbc7134b63af24) |
| [codex__annotated-single-file](codex__annotated-single-file/) | `codex/annotated-single-file` | [`5856d8c6b444`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/5856d8c6b444ba2caf3c91b40181a45321e8ac94) |
| [codex__batch-images-multi-explainer](codex__batch-images-multi-explainer/) | `codex/batch-images-multi-explainer` | [`e712b316c8c2`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/e712b316c8c28f5f4d950a7d50555fd36eeb11db) |
| [codex__headless-single-file](codex__headless-single-file/) | `codex/headless-single-file` | [`aa9e42cb0c02`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/aa9e42cb0c02683a18a0009cc080059191a2093b) |
| [codex__lime-decisions-notebook](codex__lime-decisions-notebook/) | `codex/lime-decisions-notebook` | [`b21130955cb9`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/b21130955cb9f92d738246b61804fb799c3a0f22) |
| [codex__notebook-collaboration](codex__notebook-collaboration/) | `codex/notebook-collaboration` | [`e58dc3a4afd9`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/e58dc3a4afd99bb0a954b0ba9353b797820aa8a1) |
| [main](main/) | `main` | [`65c8fa97c142`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/65c8fa97c142037c6ad15cfe046d2afeeacf1230) |
| [zain_demo](zain_demo/) | `zain_demo` | [`9d267fc4209d`](https://github.com/Seyi-Daniel/agentic-colorlime-xai/commit/9d267fc4209d767963c46f6ece50f4bd9afd7821) |

## Using a historical version

Folders are historical source snapshots, not actively maintained copies. Their setup instructions may assume the folder is the repository root.
To recreate a historical branch in a separate clone, check out its original commit from the table:

```bash
git clone --branch archive-snapshots https://github.com/Seyi-Daniel/agentic-colorlime-xai.git agentic-lime-history
cd agentic-lime-history
git switch --detach <original-commit-sha>
```

Original branch names are recorded in [manifest.json](manifest.json), together with full commit and tree identifiers.
The former notebook workflow is in [codex__notebook-collaboration](codex__notebook-collaboration/).
The previous main is in [main](main/). The collaborator branch is in [zain_demo](zain_demo/).

Historical pull requests: [#2](https://github.com/Seyi-Daniel/agentic-colorlime-xai/pull/2) and [#3](https://github.com/Seyi-Daniel/agentic-colorlime-xai/pull/3).
