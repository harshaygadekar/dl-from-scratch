#!/usr/bin/env python3
"""
Visualization Tools for Topic 02: Autograd Engine

Interactive visualizations to help understand:
- Computational graphs
- Forward and backward passes
- Gradient flow
"""

import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent / "solutions"))

try:
    from level02_vectorized import Value
except ImportError:
    try:
        from level01_naive import Value
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))
        from autograd_stub import Value


def trace(root):
    """Build a set of all nodes and edges in a computational graph."""
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_graph_ascii(root):
    """Draw a simple ASCII representation of the computational graph."""
    nodes, edges = trace(root)
    
    # Assign levels (distance from root)
    levels = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child, parent in edges:
            if parent == current and child not in levels:
                levels[child] = levels[current] + 1
                queue.append(child)
    
    # Group by level
    max_level = max(levels.values()) if levels else 0
    level_nodes = {i: [] for i in range(max_level + 1)}
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    # Print graph
    print("\n" + "="*50)
    print("Computational Graph (ASCII)")
    print("="*50)
    
    for level in range(max_level + 1):
        print(f"\nLevel {level}:")
        for node in level_nodes[level]:
            op = node._op if node._op else "input"
            parent_info = f"← from {len(node._prev)} parents" if node._prev else "(leaf)"
            print(f"  [{op}] data={node.data:.4f}, grad={node.grad:.4f} {parent_info}")
    
    print("\n" + "="*50)


def draw_graph_graphviz(root, filename="computation_graph"):
    """
    Generate a DOT file for Graphviz visualization.
    
    To view:
        dot -Tpng computation_graph.dot -o computation_graph.png
    """
    nodes, edges = trace(root)
    
    dot_lines = ['digraph G {', '  rankdir=BT;']  # Bottom to top
    
    # Add nodes
    for n in nodes:
        label = f'{n._op if n._op else "input"}\\ndata={n.data:.4f}\\ngrad={n.grad:.4f}'
        color = "lightblue" if n._prev else "lightgreen"  # Inputs are green
        if n is root:
            color = "lightyellow"  # Output is yellow
        dot_lines.append(f'  "{id(n)}" [label="{label}", style=filled, fillcolor={color}];')
    
    # Add edges
    for child, parent in edges:
        dot_lines.append(f'  "{id(child)}" -> "{id(parent)}";')
    
    dot_lines.append('}')
    
    dot_content = '\n'.join(dot_lines)
    
    # Save to file
    with open(f'{filename}.dot', 'w') as f:
        f.write(dot_content)
    
    print(f"\nGraphviz DOT file saved to: {filename}.dot")
    print("Generate image with: dot -Tpng {}.dot -o {}.png".format(filename, filename))
    
    return dot_content


def visualize_forward_backward(expression_fn, input_values, input_names):
    """
    Step-by-step visualization of forward and backward passes.
    
    Args:
        expression_fn: Function that takes Values and returns output Value
        input_values: List of float values for inputs
        input_names: List of names for inputs
    """
    print("\n" + "="*60)
    print("Forward and Backward Pass Visualization")
    print("="*60)
    
    # Create inputs
    inputs = {name: Value(val) for name, val in zip(input_names, input_values)}
    
    print("\n📥 INPUTS:")
    for name, v in inputs.items():
        print(f"  {name} = {v.data}")
    
    print("\n➡️ FORWARD PASS:")
    output = expression_fn(*inputs.values())
    print(f"  Output = {output.data}")
    
    print("\n⬅️ BACKWARD PASS:")
    output.backward()
    
    print("\n📊 GRADIENTS (∂output/∂input):")
    for name, v in inputs.items():
        print(f"  ∂/∂{name} = {v.grad}")
    
    # ASCII graph
    draw_graph_ascii(output)
    
    return output, inputs


def gradient_flow_animation():
    """Text-based animation of gradient flow."""
    print("\n" + "="*60)
    print("Gradient Flow Animation")
    print("="*60)
    
    # Expression: z = x * y + x
    print("\nExpression: z = x * y + x")
    print("Let x = 2.0, y = 3.0")
    
    frames = [
        """
FORWARD PASS:
                  ┌───────┐
    x=2.0 ──────▶│   *   │──▶ a=6.0
                  └───────┘
    y=3.0 ──────▶│       │
                  └───────┘
    
    a=6.0 ──────▶┌───────┐
                  │   +   │──▶ z=8.0
    x=2.0 ──────▶│       │
                  └───────┘
        """,
        """
BACKWARD PASS (Step 1 - Start at output):
    
    z.grad = 1.0  (∂z/∂z = 1)
        """,
        """
BACKWARD PASS (Step 2 - Through +):
    
    z = a + x
    ∂z/∂a = 1, ∂z/∂x = 1
    
    a.grad += 1.0 * z.grad = 1.0
    x.grad += 1.0 * z.grad = 1.0  (first contribution)
        """,
        """
BACKWARD PASS (Step 3 - Through *):
    
    a = x * y
    ∂a/∂x = y = 3, ∂a/∂y = x = 2
    
    x.grad += y * a.grad = 3 * 1.0 = 3.0  (second contribution)
    y.grad += x * a.grad = 2 * 1.0 = 2.0
        """,
        """
FINAL GRADIENTS:
    
    x.grad = 1.0 + 3.0 = 4.0  ✓ (combined from both paths)
    y.grad = 2.0              ✓
    
    Interpretation:
    - If we change x by ε, z changes by 4ε
    - If we change y by ε, z changes by 2ε
        """
    ]
    
    for i, frame in enumerate(frames):
        print(f"\n{'─'*40}")
        print(frame)
        if i < len(frames) - 1:
            input("Press Enter to continue...")


def main():
    """Run all visualizations."""
    print("\n" + "="*60)
    print("Topic 02: Autograd Visualization Demo")
    print("="*60)
    
    # Simple expression
    def expr1(x, y):
        return x * y + x
    
    output, inputs = visualize_forward_backward(
        expr1,
        [2.0, 3.0],
        ['x', 'y']
    )
    
    # More complex expression
    print("\n\n" + "="*60)
    print("Complex Expression")
    print("="*60)
    
    def expr2(x, y):
        a = x + y          # 5
        b = x - y          # -1
        c = a * b          # -5
        d = c * c          # 25
        return d
    
    output, inputs = visualize_forward_backward(
        expr2,
        [2.0, 3.0],
        ['x', 'y']
    )
    
    # Generate Graphviz
    print("\n\nGenerating Graphviz file...")
    draw_graph_graphviz(output, "complex_graph")
    
    # Animation
    print("\n\nWould you like to see the gradient flow animation? (y/n)")
    try:
        if input().lower() == 'y':
            gradient_flow_animation()
    except EOFError:
        pass
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
