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
    'CIRCLE',
    'SQUARE',
    'RECTANGLE',
    'OVAL',
    'RHOMBUS',
    'TRIANGLE',
    'POLYGON'
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