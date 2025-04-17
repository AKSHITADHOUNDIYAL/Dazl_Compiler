import re
import sys
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
import numpy as np
from strings_with_arrows import *
import string

# CONSTANTS
DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

# ERRORS
class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)      

# TOKENS
TT_DRAW      = 'DRAW'
TT_SHAPE     = 'SHAPE'
TT_INT       = 'INT'
TT_COMMA     = 'COMMA'
TT_LPAREN    = 'LPAREN'
TT_RPAREN    = 'RPAREN'
TT_EOF       = 'EOF'

SHAPE_KEYWORDS = [
    'CUBE',
    'SPHERE',
    'CUBOID',
    'CYLINDER',
    'CONE',
    'PYRAMID'
]

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        if self.value is not None:
            return f'{self.type}:{self.value}'
        return f'{self.type}'

# Lexical Analysis
class Lexer:
    def __init__(self, code):
        self.code = code
        self.pos = 0
        self.current_char = self.code[self.pos] if self.code else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.code):
            self.current_char = self.code[self.pos]
        else:
            self.current_char = None

    def make_tokens(self):
        tokens = []
        while self.current_char is not None:
            if self.current_char in ' \t\n':
                self.advance()
            elif self.current_char.isdigit():
                tokens.append(self.make_number())
            elif self.current_char.isalpha():
                tokens.append(self.make_identifier())
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            else:
                char = self.current_char
                self.advance()
                raise Exception(f"Illegal character: '{char}' at position {self.pos}")
        
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens

    def make_number(self):
        num_str = ''
        pos_start = self.pos
        while self.current_char is not None and self.current_char.isdigit():
            num_str += self.current_char
            self.advance()
        return Token(TT_INT, int(num_str), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos
        while self.current_char is not None and self.current_char.isalnum():
            id_str += self.current_char.upper()
            self.advance()
        
        if id_str == 'DRAW':
            return Token(TT_DRAW, id_str, pos_start, self.pos)
        elif id_str in SHAPE_KEYWORDS:
            return Token(TT_SHAPE, id_str, pos_start, self.pos)
        else:
            raise Exception(f"Unknown identifier: '{id_str}' at position {pos_start}")
        
# AST Node
class ASTNode:
    def __init__(self, node_type, value=None, params=None):
        self.type = node_type
        self.value = value
        self.params = params if params else []

    def __repr__(self):
        return f"ASTNode({self.type}, {self.value}, {self.params})"
    
# Syntax Analysis (Parser)
def parse(tokens):
    ast = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token.type == TT_DRAW:
            if i + 1 < len(tokens) and tokens[i + 1].type == TT_SHAPE:
                shape_token = tokens[i + 1]
                i += 2
                params = []

                if i < len(tokens) and tokens[i].type == TT_LPAREN:
                    i += 1
                    while i < len(tokens) and tokens[i].type != TT_RPAREN:
                        if tokens[i].type == TT_INT:
                            params.append(tokens[i].value)
                        if tokens[i].type == TT_COMMA:
                            i += 1
                            continue
                        i += 1
                    if i < len(tokens) and tokens[i].type == TT_RPAREN:
                        i += 1
                ast.append(ASTNode("DRAW", shape_token.value, params))
            else:
                i += 1
        else:
            i += 1
    return ast

# Semantic Analysis
def analyze(ast):
    shape_defaults = {
        "SPHERE": [100],
        "CUBE": [100],
        "CUBOID": [100, 50, 30],
        "CYLINDER": [30, 70],
        "CONE": [30, 70],
        "PYRAMID": [60, 80]
    }

    for node in ast:
        shape = node.value.upper()
        expected = shape_defaults.get(shape, [])
        if not node.params or len(node.params) < len(expected):
            print(f"Warning: {shape} expects {len(expected)} parameters, using default values.")
            node.params = expected
    return ast

#  IR Generation
def generate_ir(ast):
    ir = []
    for node in ast:
        ir.append(ASTNode("DRAW", node.value, node.params))
    return ir

# IR Optimization
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


# Assembly Code Generation
def generate_assembly(ir):
    assembly = []
    for node in ir:
        params_str = ', '.join(map(str, node.params))
        assembly.append(f"{node.type} {node.value} {params_str}")
    return assembly


# Machine Code Simulation
def generate_machine_code(assembly):
    machine_code = [f"0x{abs(hash(line)) % (10**8):08X}" for line in assembly]
    return machine_code


# Execution/Rendering
def render_3d(ir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_zlim([0, 1000])
    ax.axis('off')

    origin_x, origin_y, origin_z = 50, 50, 50
    spacing = 250

    for i, node in enumerate(ir):
        shape, params = node.value, node.params
        x = origin_x + (i % 2) * spacing
        y = origin_y + ((i // 2) % 2) * spacing
        z = origin_z + (i // 4) * spacing

        if shape == "SPHERE":
            r = params[0]
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            xs = x + r * np.cos(u) * np.sin(v)
            ys = y + r * np.sin(u) * np.sin(v)
            zs = z + r * np.cos(v)
            ax.plot_wireframe(xs, ys, zs, color="b")

        elif shape == "CUBE":
            s = params[0] / 2
            vertices = [
                (x - s, y - s, z - s), (x + s, y - s, z - s),
                (x + s, y + s, z - s), (x - s, y + s, z - s),
                (x - s, y - s, z + s), (x + s, y - s, z + s),
                (x + s, y + s, z + s), (x - s, y + s, z + s)
            ]
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]
            for start, end in edges:
                ax.plot(*zip(vertices[start], vertices[end]), color='r')

        elif shape == "CUBOID":
            w, h, d = [val / 2 for val in params]
            vertices = [
                (x - w, y - h, z - d), (x + w, y - h, z - d),
                (x + w, y + h, z - d), (x - w, y + h, z - d),
                (x - w, y - h, z + d), (x + w, y - h, z + d),
                (x + w, y + h, z + d), (x - w, y + h, z + d)
            ]
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]
            for start, end in edges:
                ax.plot(*zip(vertices[start], vertices[end]), color='g')

        elif shape == "CYLINDER":
            if len(params) >= 2:
                r, h = params[0], params[1]
                z_vals = np.linspace(0, h, 30)
                theta = np.linspace(0, 2*np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_vals)
                x_grid = x + r * np.cos(theta_grid)
                y_grid = y + r * np.sin(theta_grid)
                z_grid += z
                ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='c')

        elif shape == "CONE":
            if len(params) >= 2:
                r, h = params[0], params[1]
                z_vals = np.linspace(0, h, 30)
                theta = np.linspace(0, 2*np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_vals)
                radius_grid = r * (1 - z_grid / h)
                x_grid = x + radius_grid * np.cos(theta_grid)
                y_grid = y + radius_grid * np.sin(theta_grid)
                z_grid += z
                ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='orange')

        elif shape == "PYRAMID":
            if len(params) >= 2:
                b, h = params[0], params[1]  # Use only first two
                base = [
                    (x - b/2, y - b/2, z), (x + b/2, y - b/2, z),
                    (x + b/2, y + b/2, z), (x - b/2, y + b/2, z)
                ]
                apex = (x, y, z + h)
                for i in range(4):
                    ax.plot(*zip(base[i], base[(i + 1) % 4]), color='m')
                    ax.plot(*zip(base[i], apex), color='m')
            else:
                print(f"Invalid number of parameters for PYRAMID: {params}")
                continue

    plt.show()

def render(ir):
    is_3d = any(node.value in {"SPHERE", "CUBE", "CUBOID", "CYLINDER", "CONE", "PYRAMID"} for node in ir)
    if is_3d:
        render_3d(ir)

# Full Compiler Pipeline
def compile_dazl(code):
    lexer = Lexer(code)
    tokens = lexer.make_tokens()
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

    filename = sys.argv[1]
    with open(filename, 'r') as f:
        code = f.read()
    
    compile_dazl(code)