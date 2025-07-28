# archivo: view_detail_text_cattree.py

from sklearn.tree import _tree

def view_text_cattree(tree, feature_names):
    tree_ = tree.tree_
    impurity_label = "Entropy" if tree.criterion == "entropy" else "Impurity"  # nuevo
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
            values = tree_.value[node][0]
            pred_class = values.argmax()
            samples = tree_.n_node_samples[node]
            impurity = tree_.impurity[node]
            pct = (samples / total_samples) * 100
            return (
                f"{indent}Predicted Class: {pred_class}, "
                f"Samples: {samples} ({pct:.1f}%), "
                f"{impurity_label}: {impurity:.4f}\n"  # modificado
            )
    return recurse(0, 0)

def view_text_cattree_detailed(tree, feature_names):
    tree_ = tree.tree_
    impurity_label = "Entropy" if tree.criterion == "entropy" else "Impurity"  # nuevo
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    total_samples = tree_.n_node_samples[0]
    def recurse(node, depth):
        indent = "|   " * depth
        samples = tree_.n_node_samples[node]
        impurity = tree_.impurity[node]
        values = tree_.value[node][0]
        pred_class = values.argmax()
        pct = (samples / total_samples) * 100
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            line = (
                f"{indent}|--- {name} <= {threshold:.2f} "
                f"[Samples: {samples} ({pct:.1f}%), {impurity_label}: {impurity:.4f}]\n"  # modificado
            )
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return line + left + f"{indent}|--- {name} >  {threshold:.2f} " \
                f"[Samples: {samples} ({pct:.1f}%), {impurity_label}: {impurity:.4f}]\n" + right  # modificado
        else:
            return (
                f"{indent}Predicted Class: {pred_class}, "
                f"Samples: {samples} ({pct:.1f}%), "
                f"{impurity_label}: {impurity:.4f}\n"  # modificado
            )
    return recurse(0, 0)

def view_text_cattree_pruned(tree, feature_names, min_pct=0.5):
    tree_ = tree.tree_
    impurity_label = "Entropy" if tree.criterion == "entropy" else "Impurity"  # nuevo
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    total_samples = tree_.n_node_samples[0]
    def recurse(node, depth):
        indent = "|   " * depth
        samples = tree_.n_node_samples[node]
        impurity = tree_.impurity[node]
        values = tree_.value[node][0]
        pred_class = values.argmax()
        pct = (samples / total_samples) * 100
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if pct < min_pct:
                return (
                    f"{indent}|--- {name} <= {threshold:.2f} [Low_N ({pct:.1f}%)] (--)\n"
                    f"{indent}Predicted Class: {pred_class}, [Low_N ({pct:.1f}%)] (--)\n"
                )
            line = (
                f"{indent}|--- {name} <= {threshold:.2f} "
                f"[Samples: {samples} ({pct:.1f}%), {impurity_label}: {impurity:.4f}]\n"  # modificado
            )
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return line + left + f"{indent}|--- {name} >  {threshold:.2f} " \
                f"[Samples: {samples} ({pct:.1f}%), {impurity_label}: {impurity:.4f}]\n" + right  # modificado
        else:
            if pct < min_pct:
                return f"{indent}Predicted Class: {pred_class}, [Low_N ({pct:.1f}%)] (--)\n"
            return (
                f"{indent}Predicted Class: {pred_class}, "
                f"Samples: {samples} ({pct:.1f}%), "
                f"{impurity_label}: {impurity:.4f}\n"  # modificado
            )
    return recurse(0, 0)