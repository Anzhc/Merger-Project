# Node Development Guidelines for Model Merger 2

This document explains how to implement new nodes for the **Model_merger_2** project. Nodes are small Python modules discovered automatically at runtime. Each node exposes an operation that can be wired into a graph and executed by the application.

## Directory Structure

All node modules live under `Model_merger_2/nodes`. Folders inside this directory are used as categories. For example:

```
Model_merger_2/nodes/
├── merge_methods/
│   └── simple_interpolation.py
├── model_loading/
│   └── load_model.py
├── model_saving/
│   └── save_model.py
└── utils.py
```

Any `.py` file placed somewhere within `nodes` (except those starting with `_`) will be imported when the server starts.

## Required Elements

Each node module must define the following:

1. **`NODE_TYPE`** – a unique string identifier. The prefix before the slash is treated as the category, e.g. `"model_loading/load_model"`.
2. **`NODE_CATEGORY`** – user facing label used to group nodes in the editor sidebar, such as `"Merge method"` or `"Model Loading"`.
3. **`execute(node, inputs)`** – a function that receives the node JSON and a list of input values. It must return the value that represents the node's output.
4. **`get_spec()`** – returns a dictionary describing the node's UI in the editor. This specification is used by `static/index.html` to build the LiteGraph node class on the client side.

A helper function `get_params` is available in `nodes/utils.py` to extract properties stored on a node:

```python
from ..utils import get_params
params = get_params(node)
```

## Specification Format

`get_spec()` should return a dictionary with the following keys:

- `type` (string): must match `NODE_TYPE`.
- `title` (string): human readable name shown in the editor sidebar.
- `category` (string): optional; defaults to the text before the `/` in `type`.
- `node_category` (string, optional): human friendly label shown in the sidebar. Defaults to `category` if omitted.
- `inputs` (list): definitions of input slots. Each item is `{"name": <str>, "type": <str>}`.
- `outputs` (list): definitions of output slots in the same format as inputs.
- `widgets` (list): UI elements displayed on the node. Supported kinds in the current implementation are:
  - **`button`** – `{ "kind": "button", "name": "Label", "action": "<endpoint>", "assignTo": "<property>" }`
    - Performs a request to `action` when clicked. The returned `path` or `value` is assigned to the property named by `assignTo`.
  - **`text`** – `{ "kind": "text", "name": "Label", "bind": "<property>", "options": {...} }`
    - Shows an editable text field bound to a property.
  - **`slider`** – `{ "kind": "slider", "name": "Label", "bind": "<property>", "options": {"min": 0, "max": 1, "step": 0.01} }`
    - Displays a slider widget.
  Additional widget kinds can be added by extending `static/index.html`:
  - **`number`** – numeric input field.
  - **`checkbox`** – boolean toggle widget.
  - **`combo`** – drop-down selection list.
  - **`color`** – color picker widget.
  - **`textarea`** – multiline text field.
- `properties` (dict): default property values for new instances of the node.
- `help` (string, optional): description shown as a tooltip in the editor.
- `tooltip` (string, optional): text displayed when hovering over the node.
- `size` (list, optional): [width, height] defining the node's default dimensions.
- `color` (string, optional): CSS color used for the node background.
  

### Example

A minimal node might look like this:

```python
from ..utils import get_params

NODE_TYPE = 'example/sample'
NODE_CATEGORY = 'Example'

def execute(node, inputs):
    params = get_params(node)
    # perform operation using params and inputs
    return params.get('value')


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Sample Node',
        'category': 'example',
        'inputs': [],
        'outputs': [{'name': 'out', 'type': 'value'}],
        'widgets': [
            {'kind': 'text', 'name': 'Value', 'bind': 'value'},
        ],
        'properties': {'value': ''},
        'tooltip': 'Displays the provided value',
    }
```

Save the file as `Model_merger_2/nodes/example/sample.py`. When the server starts, `app.py` will import it automatically and the new node will appear in the editor under the **Example** category header.

