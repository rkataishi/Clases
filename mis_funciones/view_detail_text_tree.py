# archivo: view_detail_text_tree.py

from sklearn.tree import _tree

def view_text_tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    total_samples = tree_.n_node_samples[0]
    def recurse(node, depth):
        indent = "|   " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            line = f"{indent}|--- {name} <= {threshold:.2f}\n"
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return line + left + f"{indent}|--- {name} >  {threshold:.2f}\n" + right
        else:
            value = tree_.value[node][0][0]
            samples = tree_.n_node_samples[node]
            impurity = tree_.impurity[node]
            pct = (samples / total_samples) * 100
            return (
                f"{indent}Predicted Value: [{value:.2f}], "
                f"Samples: {samples} ({pct:.1f}%), "
                f"Error_MSE: {impurity:.2f}\n"
            )
    return recurse(0, 0)

def view_text_tree_detailed(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    total_samples = tree_.n_node_samples[0]
    def recurse(node, depth):
        indent = "|   " * depth
        samples = tree_.n_node_samples[node]
        impurity = tree_.impurity[node]
        value = tree_.value[node][0][0]
        pct = (samples / total_samples) * 100
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            line = (
                f"{indent}|--- {name} <= {threshold:.2f} "
                f"[Samples: {samples} ({pct:.1f}%), Error_MSE: {impurity:.2f}]\n"
            )
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return line + left + f"{indent}|--- {name} >  {threshold:.2f} " \
                f"[Samples: {samples} ({pct:.1f}%), Error_MSE: {impurity:.2f}]\n" + right
        else:
            return (
                f"{indent}Predicted Value: [{value:.2f}], "
                f"Samples: {samples} ({pct:.1f}%), "
                f"Error_MSE: {impurity:.2f}\n"
            )
    return recurse(0, 0)

def view_text_tree_pruned(tree, feature_names, min_pct=0.5):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    total_samples = tree_.n_node_samples[0]
    def recurse(node, depth):
        indent = "|   " * depth
        samples = tree_.n_node_samples[node]
        impurity = tree_.impurity[node]
        value = tree_.value[node][0][0]
        pct = (samples / total_samples) * 100
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if pct < min_pct:
                return (
                    f"{indent}|--- {name} <= {threshold:.2f} [Low_N ({pct:.1f}%)] (--)\n"
                    f"{indent}Predicted Value: [{value:.2f}], [Low_N ({pct:.1f}%)] (--)\n"
                )
            line = (
                f"{indent}|--- {name} <= {threshold:.2f} "
                f"[Samples: {samples} ({pct:.1f}%), Error_MSE: {impurity:.2f}]\n"
            )
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return line + left + f"{indent}|--- {name} >  {threshold:.2f} " \
                f"[Samples: {samples} ({pct:.1f}%), Error_MSE: {impurity:.2f}]\n" + right
        else:
            if pct < min_pct:
                return f"{indent}Predicted Value: [{value:.2f}], [Low_N ({pct:.1f}%)] (--)\n"
            return (
                f"{indent}Predicted Value: [{value:.2f}], "
                f"Samples: {samples} ({pct:.1f}%), "
                f"Error_MSE: {impurity:.2f}\n"
            )
    return recurse(0, 0)

def extract_sorted_tree_paths(tree, feature_names, min_pct=0.5):
    tree_ = tree.tree_
    total_samples = tree_.n_node_samples[0]
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    valid_paths = []
    low_n_nodes = []
    def recurse(node, path):
        samples = tree_.n_node_samples[node]
        pct = (samples / total_samples) * 100
        impurity = tree_.impurity[node]
        value = tree_.value[node][0][0]
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_desc = f"{name} <= {threshold:.2f} (N: {pct:.1f}%, MSE: {impurity:.2f})"
            right_desc = f"{name} > {threshold:.2f} (N: {pct:.1f}%, MSE: {impurity:.2f})"
            recurse(tree_.children_left[node], path + [left_desc])
            recurse(tree_.children_right[node], path + [right_desc])
        else:
            if pct < min_pct:
                low_n_nodes.append({
                    "value": value,
                    "pct": pct
                })
            else:
                valid_paths.append({
                    "value": value,
                    "samples": samples,
                    "pct": pct,
                    "mse": impurity,
                    "path": list(reversed(path))
                })
    recurse(0, [])
    low_n_sorted = sorted(low_n_nodes, key=lambda x: -x["value"])
    valid_sorted = sorted(valid_paths, key=lambda x: -x["value"])
    lines = ["\n=== PATHS DEL DECISION TREE, ORDENADOS POR PREDICCION ===\n"]
    for entry in low_n_sorted:
        lines.append(f"Predicted Value: [{entry['value']:.2f}] --> N: {entry['pct']:.1f}%")
    for entry in valid_sorted:
        path_str = " --> ".join(entry['path'])
        lines.append(
            f"\nPredicted Value: [{entry['value']:.2f}]\n"
            f"N: {entry['samples']} ({entry['pct']:.1f}%), MSE: {entry['mse']:.2f} --> {path_str}"
        )
    return "\n".join(lines)