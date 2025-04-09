import re
import sys
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
import numpy as np

# Phase 1: Lexical Analysis
def tokenize(code):
    return [token.upper() for token in re.findall(r'\b(DRAW|CIRCLE|SQUARE|RECTANGLE|OVAL|RHOMBUS|TRIANGLE|POLYGON)\b|\d+|\(|\)|,', code, re.IGNORECASE) if token.strip()]

# AST Node
class ASTNode:
    def __init__(self, node_type, value=None, params=None):
        self.type = node_type
        self.value = value
        self.params = params if params else []

    def __repr__(self):
        return f"ASTNode({self.type}, {self.value}, {self.params})"
    
# Phase 2: Syntax Analysis (Parser)
def parse(tokens):
    ast = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "DRAW" and i + 1 < len(tokens):
            shape = tokens[i + 1]
            params = []
            i += 2
            if i < len(tokens) and tokens[i] == "(":
                i += 1
                while i < len(tokens) and tokens[i] != ")":
                    if tokens[i].isdigit():
                        params.append(int(tokens[i]))
                    i += 1
                if i < len(tokens) and tokens[i] == ")":
                    i += 1
            ast.append(ASTNode("DRAW", shape, params))
        else:
            i += 1
    return ast


# Phase 3: Semantic Analysis
def analyze(ast):
    shape_defaults = {
        "CIRCLE": [50],
        "SQUARE": [50],
        "RECTANGLE": [100, 50],
        "OVAL": [80, 40],
        "RHOMBUS": [60, 60],
        "TRIANGLE": [80, 60],
        "POLYGON": [60, 60, 60]
    }

    for node in ast:
        shape = node.value
        params = node.params
        expected = shape_defaults.get(shape, [])

        if shape == "POLYGON":
            if len(params) < 3:
                print(f"Semantic Error: POLYGON requires at least 3 sides.")
                return None
        elif len(params) < len(expected):
            print(f"Warning: {shape} expects {len(expected)} parameters, using default values.")
            params += expected[len(params):]

        node.params = params
    return ast

# Phase 4:  IR Generation
def generate_ir(ast):
    ir = []
    for node in ast:
        ir.append(ASTNode("DRAW", node.value, node.params))
    return ir

# Phase 5: IR Optimization
def optimize_ir(ast):
    optimized = []
    seen = set()

    for node in ast:
        identifier = (node.type, node.value, tuple(node.params))
        if identifier not in seen:
            seen.add(identifier)
            optimized.append(node)
        else:
            print(f"Optimization: Removed duplicate {node.value} with params {node.params}")

    return optimized


# Phase 6: Assembly Code Generation
def generate_assembly(ir):
    assembly = []
    for node in ir:
        params_str = ', '.join(map(str, node.params))
        assembly.append(f"{node.type} {node.value} {params_str}")
    return assembly


# Phase 7: Machine Code Simulation
def generate_machine_code(assembly):
    machine_code = [f"0x{abs(hash(line)) % (10**8):08X}" for line in assembly]
    return machine_code


# Phase 8: Execution/Rendering
def render(ir):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.set_aspect('equal')
    ax.axis('off')

    positions = [(100, 500), (300, 500), (500, 500),
                 (100, 300), (300, 300), (500, 300),
                 (100, 100), (300, 100), (500, 100)]
    pos_index = 0

    for node in ir:
        if pos_index >= len(positions):
            break
        shape, params = node.value, node.params
        center_x, center_y = positions[pos_index]
        pos_index += 1

        if shape == "CIRCLE":
            ax.add_patch(patches.Circle((center_x, center_y), params[0], fill=False))
        elif shape == "SQUARE":
            side = params[0]
            ax.add_patch(patches.Rectangle((center_x - side/2, center_y - side/2), side, side, fill=False))
        elif shape == "RECTANGLE":
            width, height = params
            ax.add_patch(patches.Rectangle((center_x - width/2, center_y - height/2), width, height, fill=False))
        elif shape == "OVAL":
            width, height = params
            ax.add_patch(patches.Ellipse((center_x, center_y), width, height, fill=False))
        elif shape == "RHOMBUS":
            d1, d2 = params
            verts = [
                (center_x, center_y + d2 / 2),
                (center_x + d1 / 2, center_y),
                (center_x, center_y - d2 / 2),
                (center_x - d1 / 2, center_y),
                (center_x, center_y + d2 / 2)
            ]
            codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
            ax.add_patch(patches.PathPatch(Path(verts, codes), fill=False))
        elif shape == "TRIANGLE":
            base, height = params
            verts = [
                (center_x, center_y + height / 2),
                (center_x + base / 2, center_y - height / 2),
                (center_x - base / 2, center_y - height / 2),
                (center_x, center_y + height / 2)
            ]
            codes = [Path.MOVETO] + [Path.LINETO]*2 + [Path.CLOSEPOLY]
            ax.add_patch(patches.PathPatch(Path(verts, codes), fill=False))
        elif shape == "POLYGON":
            sides = len(params)
            angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
            verts = [(center_x + params[i] * np.cos(angle), center_y + params[i] * np.sin(angle)) for i, angle in enumerate(angles)]
            verts.append(verts[0])
            codes = [Path.MOVETO] + [Path.LINETO] * (sides - 1) + [Path.CLOSEPOLY]
            ax.add_patch(patches.PathPatch(Path(verts, codes), fill=False))

    plt.gca().invert_yaxis()
    plt.show()


# Full Compiler Pipeline
def compile_dazl(code):
    tokens = tokenize(code)
    ast = parse(tokens)
    sem_ast = analyze(ast)

    if not sem_ast:
        print("Compilation failed due to semantic error.")
        return

    ir = generate_ir(sem_ast)
    optimized_ir = optimize_ir(ir)
    assembly = generate_assembly(optimized_ir)
    machine_code = generate_machine_code(assembly)

    print("\n==== TOKENS ====")
    print(tokens)

    print("\n==== AST ====")
    for node in ast:
        print(node)

    print("\n==== IR ====")
    for node in ir:
        print(node)

    print("\n==== OPTIMIZED IR ====")
    for node in optimized_ir:
        print(node)

    print("\n==== ASSEMBLY ====")
    for line in assembly:
        print(line)

    print("\n==== MACHINE CODE (SIMULATED) ====")
    for line in machine_code:
        print(line)

    print("\n==== EXECUTION OUTPUT (RENDERED SHAPES) ====")
    render(optimized_ir)

# Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dazl_compiler.py <filename.dazl>")
        sys.exit(1)

    try:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            code = f.read().strip()
            if not code:
                print("Error: File is empty!")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{sys.argv[1]}' not found.")
        sys.exit(1)

    compile_dazl(code)
