# pyTrackBridge

A non-intrusive multimodal tracking experiment platform based on pytracking.

## Design Philosophy

- **Non-intrusive**: No modification to pytracking source code; runtime extension only
- **Minimal**: End users only need to write a script — a few dozen lines of config to run multimodal experiments
- **Transparent**: All replacement points are explicitly declared in `call_tree_text`, no hidden corners
- **Flat**: No inheritance hell, no registry ceremony — just import and use directly

## Quick Start

```bash
python rig.py script/demo_combin.py
```

## Writing Experiments

```python
# script/my_exp.py
# include "core.template_bimodal_baseline"

def get_tracker_list():
    return [{
        'name': 'my_exp',
        'tracker_setting': {
            'type': 'combin',
            'infrared_params': 'dimp.dimp50',
            'visible_params': 'dimp.dimp50',
        }
    }]

def getDataset():
    return MyDataset()
```

## Extension

- **Add a tracker**: Write a class in `tracker/`, import it in your script
- **Add a dataset**: Follow the examples in `dataset/` to write a loader
- **Add a template (N-modal)**: Write a new `call_tree_text` template in `core/`

## File Structure

```
pyTrackBridge/
  rig.py              # Engine: runtime replacement
  core/               # Templates: bimodal / trimodal / fusion paradigms
  tracker/            # Fusion algorithms
  dataset/            # Dataset adapters
  script/             # Experiment configs
```

## Core Mechanism

`rig.py` reads the `call_tree_text` from a script, performs runtime function replacement on pytracking according to the manifest, and then launches the experiment.

