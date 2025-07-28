# view_dectree.py

import os
import subprocess
from sklearn.tree import export_graphviz
from IPython.display import Image, display

def view_dectree(
    tree_model,
    features,
    nombre_base='arbol_modelo',
    output_dir='model_outputs',
    horizontal_spacing=1.0,
    vertical_spacing=1.0,
    font_size=10,
    flecha_grosor_factor=2,
    min_leaf_pct=0.0  # nuevo parámetro para ocultar hojas con baja frecuencia
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
    dot_content = dot_content.replace(
        'digraph Tree {',
        f'digraph Tree {{\n  ranksep={vertical_spacing};\n  nodesep={horizontal_spacing};'
    )

    if min_leaf_pct > 0.0:
        total_samples = tree_model.tree_.n_node_samples[0]
        nodos = dot_content.split('\n')
        nodos_filtrados = []
        for linea in nodos:
            if 'label=' in linea and 'samples = ' in linea:
                inicio = linea.find('samples = ') + len('samples = ')
                fin = linea.find('\\n', inicio)
                if inicio > 0 and fin > inicio:
                    try:
                        sample_count = int(linea[inicio:fin])
                        frac = sample_count / total_samples
                        if frac < min_leaf_pct and 'class' in linea:
                            continue
                    except ValueError:
                        pass
            nodos_filtrados.append(linea)
        dot_content = '\n'.join(nodos_filtrados)

    with open(dot_path, 'w') as file:
        file.write(dot_content)

    subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)

    display(Image(filename=png_path))