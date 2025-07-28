import os
import subprocess
from sklearn.tree import export_graphviz
from IPython.display import Image, display

def view_dectree_v(
    tree_model,
    features,
    nombre_base='arbol_modelo',
    output_dir='model_outputs',
    vertical_layout=False,
    horizontal_spacing=1.0,
    vertical_spacing=1.0,
    font_size=10,
    flecha_grosor_factor=2,
    min_leaf_pct=0.0
):
    os.makedirs(output_dir, exist_ok=True)

    dot_path = os.path.join(output_dir, f"{nombre_base}.dot")
    png_path = os.path.join(output_dir, f"{nombre_base}.png")

    export_graphviz(
        tree_model,
        out_file=dot_path,
        feature_names=features,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True,
        precision=2
    )

    with open(dot_path, 'r') as file:
        dot_content = file.read()

    dot_content = dot_content.replace('fillcolor="#ffffff"', 'fillcolor="#c6dbef"')
    dot_content = dot_content.replace('fillcolor="#e5813900"', 'fillcolor="#66c2a5"')
    dot_content = dot_content.replace('shape=box', f'shape=box, fontsize={font_size}')

    graph_attrs = []
    if vertical_layout:
        graph_attrs.append('rankdir=TB;')
        graph_attrs.append(f'ranksep={vertical_spacing};')
        graph_attrs.append(f'nodesep={horizontal_spacing};')
    else:
        graph_attrs.append(f'ranksep={horizontal_spacing};')
        graph_attrs.append(f'nodesep={vertical_spacing};')

    if graph_attrs:
        attrs_string = "\n  ".join(graph_attrs)
        dot_content = dot_content.replace(
            'digraph Tree {',
            f'digraph Tree {{\n  {attrs_string}'
        )

    if min_leaf_pct > 0.0 and hasattr(tree_model, 'tree_'):
        total_samples = tree_model.tree_.n_node_samples[0]
        lines = dot_content.splitlines()
        new_lines = []
        nodes_to_hide = set()
        node_parents = {}

        for i, line in enumerate(lines):
            if '->' in line:
                parts = line.split('->')
                parent_node = parts[0].strip()
                child_node_part = parts[1].split('[')[0].strip()
                node_parents[child_node_part] = parent_node

            if 'label=' in line and 'samples = ' in line:
                node_id_match = line.strip().split(' ')[0]
                if not node_id_match.isdigit():
                    new_lines.append(line)
                    continue

                is_leaf = True

                try:
                    samples_str_start = line.find('samples = ') + len('samples = ')
                    samples_str_end = line.find('\\n', samples_str_start)
                    if samples_str_start > -1 and samples_str_end > samples_str_start :
                        sample_count = float(line[samples_str_start:samples_str_end].replace('%',''))
                        node_proportion = sample_count / 100.0

                        is_potential_leaf = ('value =' in line and '<=' not in line and '>' not in line) or ('class =' in line)
                        
                        if node_proportion < min_leaf_pct and is_potential_leaf:
                            if node_id_match != "0":
                                nodes_to_hide.add(node_id_match)

                except ValueError:
                    pass

        for line in lines:
            hide_this_line = False
            for node_id in nodes_to_hide:
                if line.strip().startswith(node_id + " ["):
                    hide_this_line = True
                    break
            if hide_this_line:
                continue

            if '->' in line:
                parts = line.split('->')
                child_node_part = parts[1].split('[')[0].strip()
                if child_node_part in nodes_to_hide:
                    hide_this_line = True
            if hide_this_line:
                continue
            
            new_lines.append(line)
        dot_content = '\n'.join(new_lines)

    with open(dot_path, 'w') as file:
        file.write(dot_content)

    try:
        subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)
        display(Image(filename=png_path))
    except subprocess.CalledProcessError as e:
        print(f"Error al generar la imagen del árbol con Graphviz: {e}")
        print("Asegúrate de que Graphviz esté instalado y en el PATH del sistema.")
        print("Puedes instalarlo con: conda install python-graphviz o sudo apt-get install graphviz")
    except FileNotFoundError:
        print("Error: El comando 'dot' de Graphviz no se encontró.")
        print("Asegúrate de que Graphviz esté instalado y en el PATH del sistema.")
        print("Puedes instalarlo con: conda install python-graphviz o sudo apt-get install graphviz")