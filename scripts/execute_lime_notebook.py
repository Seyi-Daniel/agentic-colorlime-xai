"""Execute every demo notebook cell in a fresh Jupyter kernel; never call a live API."""
import argparse
import json
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('agentic_lime_demo_executed.ipynb'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'notebooks/agentic_lime_decisions.ipynb'
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    # Require the explicit offline default; refuse a notebook switched to live mode.
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'MODE = "demo"' in cell.source:
            break
    else:
        raise RuntimeError('Expected default demo settings cell; refusing an unknown/live notebook')
    with tempfile.TemporaryDirectory(prefix='lime-notebook-') as directory:
        client = NotebookClient(notebook, timeout=180, kernel_name='python3',
                                resources={'metadata': {'path': directory}})
        client.execute()
        errors = [o for c in notebook.cells if c.cell_type == 'code'
                  for o in c.get('outputs', []) if o.output_type == 'error']
        assert not errors, errors
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == 'code')
        summaries = list(Path(directory).glob('outputs/lime-notebook/walkthrough-*/combined_batch_summary.json'))
        if len(summaries) != 1:
            raise AssertionError('Expected exactly one combined result')
        summary = json.loads(summaries[0].read_text())
        assert summary['completed_images'] == 2, summary
        assert summary['dir_denominator'] == 2, summary
        assert all(s['mode'] == 'offline_scripted_demo' for s in summary['sessions'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print('All notebook cells executed; two offline demo images completed.')


if __name__ == '__main__':
    main()
