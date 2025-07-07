from flask import Flask, request, jsonify, Response
import tkinter as tk
from tkinter import filedialog
import json
import importlib
import os
from memory_manager import MemoryManager


def load_nodes():
    """Import node modules and return execute functions and UI specs."""
    ops = {}
    specs = []
    nodes_dir = os.path.join(os.path.dirname(__file__), 'nodes')
    for root, _, files in os.walk(nodes_dir):
        for f in files:
            if not f.endswith('.py') or f.startswith('_'):
                continue
            rel = os.path.relpath(os.path.join(root, f), os.path.dirname(__file__))
            mod_name = rel.replace(os.sep, '.')[:-3]
            module = importlib.import_module(mod_name)
            node_type = getattr(module, 'NODE_TYPE', None)
            node_category = getattr(module, 'NODE_CATEGORY', None)
            func = getattr(module, 'execute', None)
            spec_func = getattr(module, 'get_spec', None)
            if node_type and callable(func):
                ops[node_type] = func
                if callable(spec_func):
                    spec = spec_func()
                    if 'type' not in spec:
                        spec['type'] = node_type
                    spec.setdefault('category', node_type.split('/')[0])
                    if node_category:
                        spec['node_category'] = node_category
                    else:
                        spec.setdefault('node_category', spec['category'])
                    specs.append(spec)
    return ops, specs


OPERATIONS, NODE_SPECS = load_nodes()

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/node_specs')
def list_node_specs():
    """Return UI specifications for all available nodes."""
    return jsonify(NODE_SPECS)


@app.route('/choose_file')
def choose_file():
    """Open native file dialog and return selected file path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.askopenfilename()
    root.destroy()
    return jsonify({'path': path})


@app.route('/choose_folder')
def choose_folder():
    """Open native folder dialog and return selected directory path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.askdirectory()
    root.destroy()
    return jsonify({'path': path})


@app.route('/choose_save_file')
def choose_save_file():
    """Open native save file dialog and return selected path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.asksaveasfilename(defaultextension='.json',
                                        filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
    root.destroy()
    return jsonify({'path': path})


@app.route('/save_graph', methods=['POST'])
def save_graph():
    """Save posted graph JSON to selected path."""
    data = request.get_json()
    path = data.get('path')
    graph = data.get('graph')
    if not path or not graph:
        return jsonify({'error': 'missing data'}), 400
    with open(path, 'w') as f:
        json.dump(graph, f)
    return jsonify({'status': 'saved'})


@app.route('/load_graph')
def load_graph():
    """Return graph JSON from a given file path."""
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'invalid path'}), 400
    with open(path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/run', methods=['POST'])
def run_graph():
    data = request.get_json()
    nodes = {n['id']: n for n in data.get('nodes', [])}

    links_raw = data.get('links', data.get('edges', {}))
    edges = []
    if isinstance(links_raw, dict):
        for link in links_raw.values():
            # serialized links are [id, origin_id, origin_slot, target_id, target_slot, type]
            if isinstance(link, list) and len(link) >= 5:
                edges.append({'from': link[1], 'to': link[3]})
    elif isinstance(links_raw, list):
        for link in links_raw:
            if isinstance(link, dict):
                src = link.get('from') or link.get('origin_id')
                dst = link.get('to') or link.get('target_id')
                if src is not None and dst is not None:
                    edges.append({'from': src, 'to': dst})
            elif isinstance(link, list) and len(link) >= 5:
                edges.append({'from': link[1], 'to': link[3]})

    # Build dependency lists
    incoming = {nid: [] for nid in nodes}
    outgoing = {nid: [] for nid in nodes}
    for e in edges:
        src = e['from']
        dst = e['to']
        if src not in nodes or dst not in nodes:
            # ignore links pointing to missing nodes
            continue
        outgoing[src].append(dst)
        incoming[dst].append(src)

    sinks = [nid for nid, outs in outgoing.items() if len(outs) == 0]
    reachable = set()
    stack = sinks[:]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(incoming[nid])

    incoming = {nid: incoming[nid] for nid in reachable}
    outgoing = {nid: [t for t in outgoing[nid] if t in reachable] for nid in reachable}

    # copy counts for topological sort so that original
    # dependencies remain intact for execution
    remaining = {nid: len(deps) for nid, deps in incoming.items()}
    queue = [nid for nid, cnt in remaining.items() if cnt == 0]
    memory = MemoryManager(reachable, outgoing)

    processed = set()
    order = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for tgt in outgoing[nid]:
            remaining[tgt] -= 1
            if remaining[tgt] == 0:
                queue.append(tgt)

    for nid in order:
        node = nodes[nid]
        op = OPERATIONS.get(node['type'])
        if not op:
            continue
        input_values = [memory.get(i) for i in incoming[nid]]
        result = op(node, input_values)
        memory.store(nid, result)

    memory.flush()
    return jsonify({'status': 'ok'})


@app.route('/run_stream', methods=['POST'])
def run_graph_stream():
    data = request.get_json()
    nodes = {n['id']: n for n in data.get('nodes', [])}

    links_raw = data.get('links', data.get('edges', {}))
    edges = []
    if isinstance(links_raw, dict):
        for link in links_raw.values():
            if isinstance(link, list) and len(link) >= 5:
                edges.append({'from': link[1], 'to': link[3]})
    elif isinstance(links_raw, list):
        for link in links_raw:
            if isinstance(link, dict):
                src = link.get('from') or link.get('origin_id')
                dst = link.get('to') or link.get('target_id')
                if src is not None and dst is not None:
                    edges.append({'from': src, 'to': dst})
            elif isinstance(link, list) and len(link) >= 5:
                edges.append({'from': link[1], 'to': link[3]})

    incoming = {nid: [] for nid in nodes}
    outgoing = {nid: [] for nid in nodes}
    for e in edges:
        src = e['from']
        dst = e['to']
        if src not in nodes or dst not in nodes:
            continue
        outgoing[src].append(dst)
        incoming[dst].append(src)

    sinks = [nid for nid, outs in outgoing.items() if len(outs) == 0]
    reachable = set()
    stack = sinks[:]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(incoming[nid])

    incoming = {nid: incoming[nid] for nid in reachable}
    outgoing = {nid: [t for t in outgoing[nid] if t in reachable] for nid in reachable}

    remaining = {nid: len(deps) for nid, deps in incoming.items()}
    queue = [nid for nid, cnt in remaining.items() if cnt == 0]
    memory = MemoryManager(reachable, outgoing)

    order = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for tgt in outgoing[nid]:
            remaining[tgt] -= 1
            if remaining[tgt] == 0:
                queue.append(tgt)

    def generate():
        for nid in order:
            node = nodes[nid]
            op = OPERATIONS.get(node['type'])
            if op:
                input_values = [memory.get(i) for i in incoming[nid]]
                result = op(node, input_values)
                memory.store(nid, result)
            yield json.dumps({'node': nid}) + '\n'
        memory.flush()
        yield json.dumps({'status': 'done'}) + '\n'

    return Response(generate(), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
