# IMPORTS
from strings_with_arrows import *
import string
import re
import sys
import numpy as np

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
TT_INT               = 'INT'
TT_STRING            = 'STRING'
TT_IDENTIFIER        = 'IDENTIFIER'
TT_KEYWORD           = 'KEYWORD'
TT_PLUS              = 'PLUS'
TT_MINUS             = 'MINUS'
TT_MUL               = 'MUL'
TT_DIV               = 'DIV'
TT_POW               = 'POW'
TT_LPAREN            = 'LPAREN'
TT_RPAREN            = 'RPAREN'
TT_EOF               = 'EOF'

KEYWORDS = [
    'VAR',
    'IF',
    'ELSE',
    'FUN',
]

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        if self.value: 
            return f'{self.type}:{self.value}'
        return f'{self.type}'

# LEXICAL ANALYZER
class Lexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.current_char = ''
        self.pos = 0
        self.advance()

    def advance(self):
        if self.pos < len(self.source_code):
            self.current_char = self.source_code[self.pos]
            self.pos += 1
        else:
            self.current_char = None

    def make_tokens(self):
        tokens = []
        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.advance()
            else:
                char = self.current_char
                self.advance()
                return [], IllegalCharError(self.pos, self.pos, f"'{char}'")
        tokens.append(Token(TT_EOF))
        return tokens, None

    def make_number(self):
        num_str = ''
        pos_start = self.pos
        while self.current_char is not None and self.current_char in DIGITS:
            num_str += self.current_char
            self.advance()
        return Token(TT_INT, int(num_str), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos
        while self.current_char is not None and self.current_char in LETTERS_DIGITS:
            id_str += self.current_char
            self.advance()
        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_string(self):
        string = ''
        pos_start = self.pos
        self.advance()
        while self.current_char != '"' and self.current_char is not None:
            string += self.current_char
            self.advance()
        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)

# ABSTRACT SYNTAX TREE (AST)
class ASTNode:
    def __init__(self, node_type, value=None, params=None):
        self.type = node_type
        self.value = value
        self.params = params if params else []

    def __repr__(self):
        return f"ASTNode({self.type}, {self.value}, {self.params})"

# PARSING
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def expr(self):
        if self.current_tok.type in [TT_INT, TT_STRING, TT_IDENTIFIER]:
            left = self.current_tok
            self.advance()
            if self.current_tok.type == TT_PLUS:
                self.advance()
                if self.current_tok.type in [TT_INT, TT_STRING, TT_IDENTIFIER]:
                    right = self.current_tok
                    return ASTNode('BINARY_OP', '+', [left, right]), None
        return None, "Syntax Error: Expected '+' operation."

# SEMANTIC ANALYSIS
class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}

    def analyze(self, node):
        if isinstance(node, ASTNode):
            return None
        return "Semantic Error: Invalid node type."

# INTERMEDIATE REPRESENTATION (IR) GENERATION
class IRNode:
    def __init__(self, ir_code):
        self.ir_code = ir_code

class IRGenerator:
    def __init__(self):
        self.ir_code = []

    def generate(self, node):
        if isinstance(node, ASTNode):
            for child in node.params:
                self.ir_code.append(f"LOAD {child.value}")
            self.ir_code.append("ADD")
        return self.ir_code

# IR OPTIMIZATION
class IROptimizer:
    def optimize(self, ir_code):
        optimized_code = []
        for code in ir_code:
            if code.startswith("LOAD"):
                value = code.split()[1]
                optimized_code.append(f"LOAD_OPT {value}")
            else:
                optimized_code.append(code)
        return optimized_code

# ASSEMBLY CODE GENERATION
class AssemblyGenerator:
    def generate(self, optimized_ir_code):
        assembly_code = []
        for code in optimized_ir_code:
            if code.startswith("LOAD_OPT"):
                assembly_code.append(f"MOV R0, {code.split()[1]}")
            elif code == "ADD":
                assembly_code.append("ADD R0, R1")
            elif code == "SUB":
                assembly_code.append('SUB R0, R1')    
        return assembly_code

# MACHINE CODE SIMULATION
class MachineCodeSimulator:
    def simulate(self, assembly_code):
        machine_code = []
        for line in assembly_code:
            if line.startswith("MOV"):
                machine_code.append(f"0x01{line.split()[2]}")
            elif line.startswith("ADD"):
                machine_code.append("0x02")
            elif line.startswith == ("SUB"):
               machine_code.append("0x03")    
            else:
                machine_code.append("0x00")       
        return machine_code
    
def execute_machine_code(machine_code):
    stack = []
    for instr in machine_code:
        if instr.startswith('0x01'):  
            value = instr[4:]         
            stack.append(value)
        elif instr == '0x02':         
            b = stack.pop()
            a = stack.pop()
            result = a + b
            stack.append(result)
        elif instr == '0x03': 
            b = stack.pop()
            a = stack.pop()
            result = a.replace(b, "")  
            stack.append(result)    
    
    if stack:
        print(f"\n Final Output: {stack[-1]}")

# Full Compiler Pipeline
class compile_dazl:
    def __init__(self, source_code):
        self.source_code = source_code
        self.lexer = Lexer(source_code)
        self.parser = None
        self.semantic_analyzer = SemanticAnalyzer()
        self.ir_generator = IRGenerator()
        self.ir_optimizer = IROptimizer()
        self.assembly_generator = AssemblyGenerator()
        self.machine_code_simulator = MachineCodeSimulator()

    def compile(self):
        print("Starting the compilation process...\n")

        print("Step 1: Lexical Analysis...")
        tokens, error = self.lexer.make_tokens()
        if error:
            print("Lexical Error:", error)
            return error
        print(f"Lexical Analysis successful. Tokens: {tokens[:]}...")

        print("\nStep 2: Parsing...")
        self.parser = Parser(tokens)
        ast, parse_error = self.parser.expr()
        if parse_error:
            print("Parsing Error:", parse_error)
            return parse_error
        print("Parsing successful. AST generated.")

        print("\nStep 3: Semantic Analysis...")
        semantic_error = self.semantic_analyzer.analyze(ast)
        if semantic_error:
            print("Semantic Error:", semantic_error)
            return semantic_error
        print("Semantic Analysis successful.")

        print("\nStep 4: IR Generation...")
        ir_code = self.ir_generator.generate(ast)
        print("IR Generation successful. Intermediate Representation:", ir_code[:5])

        print("\nStep 5: IR Optimization...")
        optimized_ir = self.ir_optimizer.optimize(ir_code)
        print("IR Optimization successful. Optimized IR:", optimized_ir[:5])

        print("\nStep 6: Assembly Code Generation...")
        assembly_code = self.assembly_generator.generate(optimized_ir)
        print("Assembly Code Generation successful. Assembly Code:", assembly_code[:5])

        print("\nStep 7: Machine Code Simulation...")
        machine_code = self.machine_code_simulator.simulate(assembly_code)
        print("Machine Code Simulation successful. Machine Code:", machine_code[:5])

        print("Executing the code...")
        execute_machine_code(machine_code)

        print("\nCompilation complete.\n")
        return machine_code

# Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dazl_compiler.py <filename.dazl>")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, 'r') as f:
        code = f.read()

    compiler = compile_dazl(code)
    machine_code = compiler.compile()