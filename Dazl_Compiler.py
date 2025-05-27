#######################################
#### IMPORTS ####
#######################################

from strings_with_arrows import *
import sys
import string
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.dates as mdates
import mplfinance as mpf
from matplotlib.patches import Ellipse
from itertools import product, combinations
import os
import ast
import networkx as nx
import base64
from matplotlib import pyplot as plt
from io import BytesIO
import io
import sys
import streamlit as st
import operator as op

#######################################
#### CONSTANTS ####
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
#### ERRORS ####
#######################################

class Error:
  def __init__(self, pos_start, pos_end, error_name, details):
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.error_name = error_name
    self.details = details

  def as_string(self):
    result  = f'{self.error_name}: {self.details}\n'
    result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

class IllegalCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
  def __init__(self, pos_start, pos_end, details=''):
    super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, 'Runtime Error', details)
    self.context = context

  def as_string(self):
    result  = self.generate_traceback()
    result += f'{self.error_name}: {self.details}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

  def generate_traceback(self):
    result = ''
    pos = self.pos_start
    ctx = self.context

    while ctx:
      result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    return 'Traceback (most recent call last):\n' + result

#######################################
#### POSITION ####
#######################################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx         
        self.ln = ln          
        self.col = col        
        self.fn = fn          
        self.ftxt = ftxt       

    def advance(self, current_char=None):
        """
        Advance the position by one character and update line/column.
        """
        self.idx += 1
        self.col += 1

        if current_char == '\n': 
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        """
        Return a copy of the current position.
        """
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

    def __repr__(self):
        return f"Position(idx={self.idx}, ln={self.ln}, col={self.col}, fn={self.fn})"

#######################################
#### TOKENS ####
#######################################

TT_INT         = 'INT'
TT_FLOAT       = 'FLOAT'
TT_STRING      = 'STRING'
TT_IDENTIFIER  = 'IDENTIFIER'
TT_KEYWORD     = 'KEYWORD'
TT_PLUS        = 'PLUS'
TT_MINUS       = 'MINUS'
TT_MUL         = 'MUL'
TT_DIV         = 'DIV'
TT_POW         = 'POW'
TT_EQ          = 'EQ'
TT_LPAREN      = 'LPAREN'
TT_RPAREN      = 'RPAREN'
TT_LSQUARE     = 'LSQUARE'
TT_RSQUARE     = 'RSQUARE'
TT_EE          = 'EE'
TT_NE          = 'NE'
TT_LT          = 'LT'
TT_GT          = 'GT'
TT_LTE         = 'LTE'
TT_GTE         = 'GTE'
TT_COMMA       = 'COMMA'
TT_ARROW       = 'ARROW'
TT_NEWLINE     = 'NEWLINE'
TT_EOF         = 'EOF'

KEYWORDS = [
  # Programming Keywords
  'VAR','AND', 'OR', 'NOT','SET',
  'IF', 'ELSE', 'FOR', 'TO',  'WHILE', 'FOREND', 'WHILEEND',
  'PRINT',

  # List operations
  'APPEND', 'REMOVE', 'LENGTH', 'DISPLAY',

  # Charts / Plots
  'BARCHART', 'COLUMNCHART', 'LINECHART', 'PIECHART', 'HISTOGRAM',
  'SCATTERPLOT', 'BUBBLECHART', 'BOXPLOT', 'HEATMAP', 'FLOWCHART'
]

SHAPE_KEYWORDS = [
    'CIRCLE', 'SQUARE', 'RECTANGLE', 'OVAL', 'TRIANGLE', 'RHOMBUS', 'POLYGON',
    'CUBE', 'SPHERE', 'CONE', 'CYLINDER', 'PYRAMID','CUBOID','SHOW'
]


class Token:
  def __init__(self, type_, value=None, pos_start=None, pos_end=None):
    self.type = type_
    self.value = value
    self.pos_start = pos_start

    if pos_start:
      self.pos_start = pos_start.copy()
      self.pos_end = pos_start.copy()
      self.pos_end.advance()

    if pos_end:
      self.pos_end = pos_end.copy()

  def matches(self, type_, value):
    return self.type == type_ and self.value == value

  def __repr__(self):
    if self.value: return f'{self.type}:{self.value}'
    return f'{self.type}'

#######################################
#### LEXER ####
#######################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        """
        Move the lexer position to the next character and update `current_char`.
        """
        self.pos.advance(self.current_char)  
        if self.pos.idx < len(self.text):
            self.current_char = self.text[self.pos.idx]  
        else:
            self.current_char = None 

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char == '#':
                self.skip_comment()
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(self.make_minus_or_arrow())
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '[':
                tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == ']':
                tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        escape_character = False
        self.advance()

        escape_characters = {'n': '\n', 't': '\t'}

        while self.current_char is not None and (self.current_char != '"' or escape_character):
            if escape_character:
                string += escape_characters.get(self.current_char, self.current_char)
                escape_character = False
            else:
                if self.current_char == '\\':
                    escape_character = True
                else:
                    string += self.current_char
            self.advance()

        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        id_str = id_str.upper()
        if id_str in KEYWORDS:
          tok_type = TT_KEYWORD
        else:
          tok_type = TT_IDENTIFIER


        return Token(tok_type, id_str.upper(), pos_start, self.pos)

    def make_minus_or_arrow(self):
        tok_type = TT_MINUS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            tok_type = TT_ARROW

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

        return None, ExpectedCharError(pos_start, self.pos, "'=' after '!'")

    def make_equals(self):
        tok_type = TT_EQ
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_EE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = TT_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_LTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_GTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def skip_comment(self):
        self.advance()
        while self.current_char != '\n' and self.current_char is not None:
            self.advance()

#######################################
#### AST NODES ####
#######################################

class ASTNode:
    def __init__(self, node_type, value=None, params=None):
        self.type = node_type
        self.value = value
        self.params = params if params else []

    def __repr__(self):
        return f"ASTNode({self.type}, {self.value}, {self.params})"


class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
        return str(self.tok.value)

class StringNode:
  def __init__(self, tok):
    self.tok = tok
    self.value = tok.value
    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f'ListNode({self.element_nodes})'

class VarAccessNode:
  def __init__(self, var_name_tok):
    self.var_name_tok = var_name_tok

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end

    def __repr__(self):
        return f'VarAssignNode({self.var_name_tok}, {self.value_node})'


class BinOpNode:
  def __init__(self, left_node, op_tok, right_node):
    self.left_node = left_node
    self.op_tok = op_tok
    self.right_node = right_node

    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end

  def __repr__(self):
    return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
  def __init__(self, op_tok, node):
    self.op_tok = op_tok
    self.node = node

    self.pos_start = self.op_tok.pos_start
    self.pos_end = node.pos_end

  def __repr__(self):
    return f'({self.op_tok}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

class ForNode:
  def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
    self.var_name_tok = var_name_tok
    self.start_value_node = start_value_node
    self.end_value_node = end_value_node
    self.step_value_node = step_value_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.body_node.pos_end

class WhileNode:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.body_node.pos_end

class CallNode:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.pos_start = self.node_to_call.pos_start

    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.node_to_call.pos_end 

class PrintNode:
    def __init__(self, tok, expr):
        self.tok = tok  
        self.expr = expr  

    def __repr__(self):
        return f"PrintNode({self.expr})"

#######################################
#### PARSE RESULT ####
#######################################

class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def register_advancement(self):
    self.last_registered_advance_count = 1
    self.advance_count += 1

  def register(self, res):
    self.last_registered_advance_count = res.advance_count
    self.advance_count += res.advance_count
    if res.error: self.error = res.error
    return res.node

  def try_register(self, res):
    if res.error:
      self.to_reverse_count = res.advance_count
      return None
    return self.register(res)

  def success(self, node):
    self.node = node
    return self

  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self

#######################################
#### PARSER ####
#######################################

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    self.advance()

  def advance(self, ):
    self.tok_idx += 1
    if self.tok_idx < len(self.tokens):
      self.current_tok = self.tokens[self.tok_idx]
    return self.current_tok

  def parse(self):
    res = self.expr()
    if not res.error and self.current_tok.type != TT_EOF:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
      ))
    return res

  ###################################

  def expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'VAR'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type != TT_IDENTIFIER:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected identifier"
        ))

      var_name = self.current_tok
      res.register_advancement()
      self.advance()

      if self.current_tok.type != TT_EQ:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '='"
        ))

      res.register_advancement()
      self.advance()
      expr = res.register(self.expr())
      if res.error: return res
      return res.success(VarAssignNode(var_name, expr))

    node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'AND'), (TT_KEYWORD, 'OR'))))

    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
      ))

    return res.success(node)

  def comp_expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'NOT'):
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    
    node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'IF', 'FOR', 'WHILE' or 'NOT'"
      ))

    return res.success(node)

  def arith_expr(self):
    return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

  def term(self):
    return self.bin_op(self.factor, (TT_MUL, TT_DIV))

  def factor(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_PLUS, TT_MINUS):
      res.register_advancement()
      self.advance()
      factor = res.register(self.factor())
      if res.error: return res
      return res.success(UnaryOpNode(tok, factor))

    return self.power()

  def power(self):
    return self.bin_op(self.call, (TT_POW, ), self.factor)

  def call(self):
    res = ParseResult()
    atom = res.register(self.atom())
    if res.error: return res

    if self.current_tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      arg_nodes = []

      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
      else:
        arg_nodes.append(res.register(self.expr()))
        if res.error:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
          ))

        while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()

          arg_nodes.append(res.register(self.expr()))
          if res.error: return res

        if self.current_tok.type != TT_RPAREN:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
          ))

        res.register_advancement()
        self.advance()
      return res.success(CallNode(atom, arg_nodes))
    return res.success(atom)

  def atom(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_INT, TT_FLOAT):
      res.register_advancement()
      self.advance()
      return res.success(NumberNode(tok))

    elif tok.type == TT_STRING:
      res.register_advancement()
      self.advance()
      return res.success(StringNode(tok))

    elif tok.type == TT_IDENTIFIER:
      res.register_advancement()
      self.advance()
      return res.success(VarAccessNode(tok))

    elif tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      expr = res.register(self.expr())
      if res.error: return res
      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
        return res.success(expr)
      else:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ')'"
        ))

    elif tok.type == TT_LSQUARE:
      list_expr = res.register(self.list_expr())
      if res.error: return res
      return res.success(list_expr)
    
    elif tok.matches(TT_KEYWORD, 'IF'):
      if_expr = res.register(self.if_expr())
      if res.error: return res
      return res.success(if_expr)

    elif tok.matches(TT_KEYWORD, 'FOR'):
      for_expr = res.register(self.for_expr())
      if res.error: return res
      return res.success(for_expr)

    elif tok.matches(TT_KEYWORD, 'WHILE'):
      while_expr = res.register(self.while_expr())
      if res.error: return res
      return res.success(while_expr)

    elif tok.matches(TT_KEYWORD, 'FUN'):
      func_def = res.register(self.func_def())
      if res.error: return res
      return res.success(func_def)

    return res.failure(InvalidSyntaxError(
      tok.pos_start, tok.pos_end,
      "Expected int, float, identifier, '+', '-', '(', '[', IF', 'FOR', 'WHILE', 'FUN'"
    ))

  def list_expr(self):
    res = ParseResult()
    element_nodes = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TT_LSQUARE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '['"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_RSQUARE:
      res.register_advancement()
      self.advance()
    else:
      element_nodes.append(res.register(self.expr()))
      if res.error:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ']', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
        ))

      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        element_nodes.append(res.register(self.expr()))
        if res.error: return res

      if self.current_tok.type != TT_RSQUARE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ']'"
        ))

      res.register_advancement()
      self.advance()

    return res.success(ListNode(
      element_nodes,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def if_expr(self):
    res = ParseResult()
    cases = []
    else_case = None

    if not self.current_tok.matches(TT_KEYWORD, 'IF'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'IF'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    expr = res.register(self.expr())
    if res.error: return res
    cases.append((condition, expr))

    while self.current_tok.matches(TT_KEYWORD, 'ELIF'):
      res.register_advancement()
      self.advance()

      condition = res.register(self.expr())
      if res.error: return res

      if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'THEN'"
        ))

      res.register_advancement()
      self.advance()

      expr = res.register(self.expr())
      if res.error: return res
      cases.append((condition, expr))

    if self.current_tok.matches(TT_KEYWORD, 'ELSE'):
      res.register_advancement()
      self.advance()

      else_case = res.register(self.expr())
      if res.error: return res

    return res.success(IfNode(cases, else_case))

  def for_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'FOR'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'FOR'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_IDENTIFIER:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected identifier"
      ))

    var_name = self.current_tok
    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_EQ:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '='"
      ))
    
    res.register_advancement()
    self.advance()

    start_value = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'TO'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'TO'"
      ))
    
    res.register_advancement()
    self.advance()

    end_value = res.register(self.expr())
    if res.error: return res

    if self.current_tok.matches(TT_KEYWORD, 'STEP'):
      res.register_advancement()
      self.advance()

      step_value = res.register(self.expr())
      if res.error: return res
    else:
      step_value = None

    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    body = res.register(self.expr())
    if res.error: return res

    return res.success(ForNode(var_name, start_value, end_value, step_value, body))

  def while_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'WHILE'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'WHILE'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    body = res.register(self.expr())
    if res.error: return res

    return res.success(WhileNode(condition, body))


  ###################################

  def bin_op(self, func_a, ops, func_b=None):
    if func_b == None:
      func_b = func_a
    
    res = ParseResult()
    left = res.register(func_a())
    if res.error: return res

    while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()
      right = res.register(func_b())
      if res.error: return res
      left = BinOpNode(left, op_tok, right)

    return res.success(left)

################################
##### SEMATIC ANALYZER #####
################################
BUILTIN_FUNCTIONS = {
    "APPEND", "REMOVE", "LENGTH", "DISPLAY",  
    "SET",
    "BARCHART", "COLUMNCHART", "LINECHART", "PIECHART", "HISTOGRAM",
    "SCATTERPLOT", "BUBBLECHART", "BOXPLOT", "HEATMAP", "FLOWCHART",  
    "CIRCLE", "SQUARE", "RECTANGLE", "OVAL", "TRIANGLE", "RHOMBUS", "POLYGON",
    "CUBE", "SPHERE", "CONE", "CYLINDER", "PYRAMID","CUBOID","SHOW" 
}

class SemanticError(Exception):
    def __init__(self, message, node=None):
        self.message = message
        self.node = node
        super().__init__(self.message)

class SymbolTable:
    def __init__(self):
        self.symbols = {}

    def declare(self, name, value_type):
        self.symbols[name] = value_type

    def lookup(self, name):
        return self.symbols.get(name, None)

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = SymbolTable()

    def analyze(self, node):
        method_name = f'analyze_{type(node).__name__}'
        method = getattr(self, method_name, self.no_analyze_method)
        return method(node)

    def no_analyze_method(self, node):
        raise Exception(f"No analyze_{type(node).__name__} method")

    def analyze_ListNode(self, node):
        for element in node.element_nodes:
            self.analyze(element)

    def analyze_NumberNode(self, node):
        return "int" if node.tok.type == TT_INT else "float"

    def analyze_StringNode(self, node):
        return "string"

    def analyze_VarAccessNode(self, node):
        var_name = node.var_name_tok.value
        var_type = self.symbol_table.lookup(var_name)
        if var_type is None:
            raise SemanticError(f"Variable '{var_name}' not defined", node)
        return var_type

    def analyze_VarAssignNode(self, node):
        var_name = node.var_name_tok.value
        expr_type = self.analyze(node.value_node)
        self.symbol_table.declare(var_name, expr_type)
        return expr_type

    def analyze_UnaryOpNode(self, node):
        val_type = self.analyze(node.node)
        if node.op_tok.type in ('PLUS', 'MINUS') and val_type not in ('int', 'float'):
            raise SemanticError(f"Unary operator '{node.op_tok.type}' not allowed on {val_type}", node)
        return val_type

    def analyze_BinOpNode(self, node):
        left = self.analyze(node.left_node)
        right = self.analyze(node.right_node)
        op = node.op_tok.type

        if op in ('PLUS', 'MINUS', 'MUL', 'DIV', 'POW'):
            if left in ('int', 'float') and right in ('int', 'float'):
                return 'float' if 'FLOAT' in (left, right) else 'int'
            elif op == 'PLUS' and left == 'string' and right == 'string':
                return 'string'
            else:
                raise SemanticError(f"Incompatible types for {op}: {left}, {right}", node)

        elif op in ('EE', 'NE', 'LT', 'GT', 'LTE', 'GTE'):
            if left == right:
                return 'bool'
            else:
                raise SemanticError(f"Cannot compare {left} with {right}", node)

        elif op in ('AND', 'OR'):
            if left == 'bool' and right == 'bool':
                return 'bool'
            else:
                raise SemanticError("AND/OR requires boolean operands", node)

        raise SemanticError(f"Unknown operator {op}", node)

    def analyze_IfNode(self, node):
        for cond, body, _ in node.cases:
            cond_type = self.analyze(cond)
            if cond_type != 'bool':
                raise SemanticError("Condition in IF must be boolean", cond)
            self.analyze(body)        
        if node.else_case:
            self.analyze(node.else_case[0])

    def analyze_PrintNode(self, node):
      print(f"Analyzing PrintNode: {node}")
      return node  

    def analyze_ForNode(self, node):
        start_type = self.analyze(node.start_value_node)
        end_type = self.analyze(node.end_value_node)
        if node.step_value_node:
            step_type = self.analyze(node.step_value_node)
            if step_type not in ('int', 'float'):
                raise SemanticError("FOR loop step must be numeric", node.step_value_node)

        if start_type not in ('int', 'float') or end_type not in ('int', 'float'):
            raise SemanticError("FOR loop start/end must be numeric", node)

        self.symbol_table.declare(node.var_name_tok.value, 'int')  
        self.analyze(node.body_node)

    def analyze_WhileNode(self, node):
        cond_type = self.analyze(node.condition_node)
        if cond_type != 'bool':
            raise SemanticError("WHILE condition must be boolean", node.condition_node)
        self.analyze(node.body_node)

    def analyze_CallNode(self, node):
        func_name = node.node_to_call.var_name_tok.value

        if func_name in BUILTIN_FUNCTIONS:
            for arg in node.arg_nodes:
                self.analyze(arg)
            return "builtin"
        else:
            raise SemanticError(f"Unknown function '{func_name}'", node)


#######################################
##### IR GENERATOR ####
#######################################

class IRInstruction:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __repr__(self):
        formatted_args = [str(arg) if not isinstance(arg, PrintNode) else f"PrintNode({arg.expr})" for arg in self.args]
        return f"{self.op} {', '.join(map(str, self.args))}"
    
BUILTIN_FUNCTIONS = {
    "APPEND": "handle_append",
    "REMOVE": "handle_remove",
    "LENGTH": "handle_length",
    "DISPLAY": "handle_display",
    "SET": "handle_set",
    "BARCHART": "handle_barchart",
    "COLUMNCHART": "handle_columnchart",
    "LINECHART": "handle_linechart",
    "PIECHART": "handle_piechart",
    "HISTOGRAM": "handle_histogram",
    "SCATTERPLOT": "handle_scatterplot",
    "BUBBLECHART": "handle_bubblechart",
    "BOXPLOT": "handle_boxplot",
    "HEATMAP": "handle_heatmap",
    "FLOWCHART": "handle_flowchart",
    "CIRCLE": "handle_circle",
    "SQUARE": "handle_square",
    "RECTANGLE": "handle_rectangle",
    "OVAL": "handle_oval",
    "TRIANGLE": "handle_triangle",
    "RHOMBUS": "handle_rhombus",
    "POLYGON": "handle_polygon",
    "CUBE": "handle_cube",
    "SPHERE": "handle_sphere",
    "CONE": "handle_cone",
    "CYLINDER": "handle_cylinder",
    "PYRAMID": "handle_pyramid",
    "CUBOID": "handle_cuboid",
    "SHOW": "handle_show",
}
    
class IRGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0
        self.ir_code = []  

    def generate(self, node):
        method = getattr(self, f'gen_{type(node).__name__}', None)
        if method:
            return method(node)  
        else:
            raise Exception(f"No IR generator for {type(node).__name__}")

    def gen_PrintNode(self, node):
        ir_instruction = IRInstruction("PRINT", node.expr)
        print(f"Generated IR: {ir_instruction}")  
        self.ir_code.append(ir_instruction)  

    def new_temp(self):
        self.temp_counter += 1
        return f't{self.temp_counter}'

    def no_gen_method(self, node):
        raise Exception(f"No IR generator for {type(node).__name__}")

    def gen_NumberNode(self, node):
        return str(node.tok.value)

    def gen_StringNode(self, node):
        temp = self.new_temp()
        instr = IRInstruction("LOAD", temp, f'"{node.value}"')
        self.instructions.append(instr)
        return temp

    def gen_ListNode(self, node):
        temp = self.new_temp()
        values = [self.generate(el) for el in node.element_nodes if el is not None]
        self.instructions.append(IRInstruction("CREATE_LIST", temp, *values))  
        return temp

    def gen_BinOpNode(self, node):
        left = self.generate(node.left_node)
        right = self.generate(node.right_node)

        op_map = {
        'PLUS': 'ADD',   
        'MINUS': 'SUB',
        'MUL': 'MUL',
        'DIV': 'DIV',
        'POW': 'POW',
        'EE': 'EQ',
        'NE': 'NE',
        'LT': 'LT',
        'GT': 'GT',
        'LTE': 'LE',
        'GTE': 'GE',
        'AND': 'AND',
        'OR': 'OR'
        }

        if isinstance(left, str) and isinstance(right, str):
            result = self.new_temp()
            self.instructions.append(IRInstruction('CONCAT', result, left, right)) 

        elif isinstance(left, list) and isinstance(right, list):
            result = self.new_temp()
            self.instructions.append(IRInstruction('CONCAT_LIST', result, left, right)) 
        else:
            op = op_map.get(node.op_tok.type, 'UNKNOWN')
            result = self.new_temp()
            self.instructions.append(IRInstruction(op, result, left, right)) 

        return result

    def gen_UnaryOpNode(self, node):
        val = self.generate(node.node)
        op = 'NEG' if node.op_tok.type == 'MINUS' else 'POS'
        result = self.new_temp()
        self.instructions.append(IRInstruction(op, result, val))
        return result

    def gen_VarAccessNode(self, node):
        return node.var_name_tok.value

    def gen_VarAssignNode(self, node):
       value = self.generate(node.value_node)
       self.instructions.append(IRInstruction("ASSIGN", node.var_name_tok.value, value))
       return node.var_name_tok.value

    def gen_IfNode(self, node):
        for condition, body, _ in node.cases:
            cond_temp = self.generate(condition)
            self.instructions.append(IRInstruction('IF', cond_temp))
            self.generate(body)
        if node.else_case:
            self.instructions.append(IRInstruction('ELSE'))
            self.generate(node.else_case[0])

    def gen_ForNode(self, node):
        start = self.generate(node.start_value_node)
        end = self.generate(node.end_value_node)
        step = self.generate(node.step_value_node) if node.step_value_node else '1'
        var = node.var_name_tok.value
        self.instructions.append(IRInstruction('FOR', var, start, end, step))
        self.generate(node.body_node)
        self.instructions.append(IRInstruction('ENDFOR'))

    def gen_WhileNode(self, node):
        cond = self.generate(node.condition_node)
        self.instructions.append(IRInstruction('WHILE', cond))
        self.generate(node.body_node)
        self.instructions.append(IRInstruction('ENDWHILE'))

    def gen_CallNode(self, node):
        func_name = node.node_to_call.var_name_tok.value
        args = [self.generate(arg) for arg in node.arg_nodes]

        handler_method_name = BUILTIN_FUNCTIONS.get(func_name)
    
        if handler_method_name:
            handler = getattr(self, handler_method_name)
            return handler(args)
        else:
            return f"CALL_{func_name}({args})"


    def handle_append(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("APPEND", result, *args))
        return result

    def handle_remove(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("REMOVE", result, *args))
        return result

    def handle_length(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("LENGTH", result, *args))
        return result

    def handle_display(self, args):
        self.instructions.append(IRInstruction("DISPLAY", *args))
        return None 

    def handle_set(self, args):
        self.instructions.append(IRInstruction("SET", *args))
        return None  

    def handle_barchart(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("BARCHART", result, *args))
        return result

    def handle_columnchart(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("COLUMNCHART", result, *args))
        return result

    def handle_linechart(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("LINECHART", result, *args))
        return result

    def handle_piechart(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("PIECHART", result, *args))
        return result

    def handle_histogram(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("HISTOGRAM", result, *args))
        return result

    def handle_scatterplot(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("SCATTERPLOT", result, *args))
        return result

    def handle_bubblechart(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("BUBBLECHART", result, *args))
        return result

    def handle_boxplot(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("BOXPLOT", result, *args))
        return result

    def handle_heatmap(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("HEATMAP", result, *args))
        return result

    def handle_flowchart(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("FLOWCHART", result, *args))
        return result

    def handle_circle(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("CIRCLE", result, *args))
        return result

    def handle_square(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("SQUARE", result, *args))
        return result

    def handle_rectangle(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("RECTANGLE", result, *args))
        return result

    def handle_oval(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("OVAL", result, *args))
        return result

    def handle_triangle(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("TRIANGLE", result, *args))
        return result

    def handle_rhombus(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("RHOMBUS", result, *args))
        return result

    def handle_polygon(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("POLYGON", result, *args))
        return result

    def handle_cube(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("CUBE", result, *args))
        return result

    def handle_sphere(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("SPHERE", result, *args))
        return result

    def handle_cone(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("CONE", result, *args))
        return result

    def handle_cylinder(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("CYLINDER", result, *args))
        return result

    def handle_pyramid(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("PYRAMID", result, *args))
        return result

    def handle_cuboid(self, args):
        result = self.new_temp()
        self.instructions.append(IRInstruction("CUBOID", result, *args))
        return result

    def handle_show(self, args):
        self.instructions.append(IRInstruction("SHOW", *args))
        return None  
    
###############################
#### IR Optimization ####
###############################
def is_number(x):
    try:
        float(x)
        return True
    except:
        return False
    
def is_string(value):
    return isinstance(value, str)

class Optimizer:
    def __init__(self, instructions):
        self.instructions = instructions

    def optimize_constant_folding(self):
      optimized_instructions = []
      for instruction in self.instructions:
        if isinstance(instruction, IRInstruction):
            if instruction.op == 'MOV' and isinstance(instruction.args[1], str) and instruction.args[1].isdigit():
                instruction.args = [instruction.args[0], int(instruction.args[1])]
            elif instruction.op in ('ADD', 'SUB', 'MUL', 'DIV', 'POW', 'CMP'):
                if isinstance(instruction.args[1], (int, float)) and isinstance(instruction.args[2], (int, float)):
                    result = eval(f"{instruction.args[1]} {instruction.op.lower()} {instruction.args[2]}")
                    optimized_instructions.append(IRInstruction('MOV', instruction.args[0], result))
                    continue
            optimized_instructions.append(instruction)
        else:
            optimized_instructions.append(instruction)  
      self.instructions = optimized_instructions

    def optimize_dead_code(self):
        used_vars = set()
        optimized_instructions = []
        for instruction in reversed(self.instructions):
            if isinstance(instruction, IRInstruction):
                if instruction.op == 'MOV' and instruction.args[0] not in used_vars:
                  continue  
                if instruction.op == 'STORE':
                  used_vars.add(instruction.args[0])
                optimized_instructions.append(instruction)
            else:
              optimized_instructions.append(instruction)
        self.instructions = list(reversed(optimized_instructions))

    def eliminate_redundant_moves(self):
        optimized_instructions = []
        for instruction in self.instructions:
            if instruction.op == 'MOV' and instruction.args[0] == instruction.args[1]:
                continue  
            optimized_instructions.append(instruction)
        self.instructions = optimized_instructions

    def optimize_common_subexpressions(self):
      expression_map = {}
      optimized_instructions = []
      for instruction in self.instructions:
        if isinstance(instruction, IRInstruction) and instruction.op in ('ADD', 'SUB', 'MUL', 'DIV', 'POW'):
            expr = (instruction.args[1], instruction.args[2])
            if expr in expression_map:
                instruction.args = [instruction.args[0], expression_map[expr]]
            else:
                expression_map[expr] = instruction.args[0]
        optimized_instructions.append(instruction)
      self.instructions = optimized_instructions

    def simplify_conditions(self):
        optimized_instructions = []
        for instruction in self.instructions:
            if instruction.op == 'CMP' and isinstance(instruction.args[1], (int, str)) and isinstance(instruction.args[2], (int, str)):
                if instruction.args[1] == instruction.args[2]:
                    optimized_instructions.append(IRInstruction('MOV', instruction.args[0], 1))
                else:
                    optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)
        self.instructions = optimized_instructions

    def optimize(self):
        """ Method to trigger the overall optimization. """
        self.optimize_ir()    

    def optimize_ir(self):
      """Method to apply various optimization techniques to IR instructions."""
      optimized = []
    
      for instr in self.instructions:
        if isinstance(instr, tuple):
            op, *args = instr
            instr = IRInstruction(op, *args)
        
        if instr.op in ('ADD', 'SUB', 'MUL', 'DIV', 'POW'):
            _, dest, left, right = instr.args
            if is_number(left) and is_number(right):
                left_val = float(left)
                right_val = float(right)
                result = {
                    'ADD': left_val + right_val,
                    'SUB': left_val - right_val,
                    'MUL': left_val * right_val,
                    'DIV': left_val / right_val if right_val != 0 else 0,
                    'POW': left_val ** right_val
                }[instr.op]
                optimized.append(IRInstruction('ASSIGN', dest, str(result))) 
            else:
                optimized.append(instr)

        elif instr.op == 'APPEND':
            if len(instr.args) == 4:  
                _, dest, list1, list2 = instr.args
                if all(is_string(item) for item in list1) and all(is_string(item) for item in list2):
                    combined_list = list1 + list2  
                    optimized.append(IRInstruction('ASSIGN', dest, str(combined_list)))  
                else:
                    optimized.append(instr)  
            else:
                optimized.append(instr)

        elif instr.op == 'REMOVE':
            if len(instr.args) == 4:
                _, dest, list1, list2 = instr.args
                if all(is_string(item) for item in list1) and all(is_string(item) for item in list2):
                    result_list = [item for item in list1 if item not in list2]
                    optimized.append(IRInstruction('ASSIGN', dest, str(result_list)))  
                else:
                    optimized.append(instr)  
            else:
                    optimized.append(instr)
        

        else:
            optimized.append(instr)  
        self.instructions = optimized


##############################
### ASSEMBLY CODE GENERATOR ###
###############################

def flatten(arg, visited=None):
    if visited is None:
        visited = set()

    if id(arg) in visited:
        return '...'

    visited.add(id(arg))

    if isinstance(arg, IRInstruction):
        return f"{arg.op}({', '.join(str(a) for a in arg.args)})"
    
    elif isinstance(arg, (tuple, list)):
        flat = []
        for a in arg:
            flat.append(flatten(a, visited))
        return flat
    
    else:
        return arg


def generate_assembly(optimizer):

    def __init__(self, ir_generator):
        self.ir_generator = ir_generator
        self.assembly = []
        self.temp_count = 0

    def new_temp(self):
        temp = f"t{self.temp_count}"
        self.temp_count += 1
        return temp
    
    ir_instructions = optimizer.instructions
    assembly = []

    for instr in ir_instructions:
        if not isinstance(instr, IRInstruction):
            assembly.append(f"# Unknown instruction format: {instr}")
            continue

        op = instr.op
        args = instr.args


        def clean_arg(arg):
            flattened = flatten(arg)
            if isinstance(flattened, list):
                return ', '.join(str(a) for a in flattened if a is not None)
            elif isinstance(flattened, str):
                return flattened.strip('"').strip('[]')
            elif flattened is None:
                return ""
            else:
                return str(flattened)

        dest = args[0] if len(args) > 0 else ""
        arg1 = args[1] if len(args) > 1 else ""
        arg2 = args[2] if len(args) > 2 else ""

        arg1_clean = clean_arg(arg1)
        arg2_clean = clean_arg(arg2)

        if op == 'MOV':
            return f"MOV {args[0]}, {args[1]}"
        
        elif op == 'ADD':
            return f"ADD {args[0]}, {args[1]}, {args[2]}"
        
        elif op == 'SUB':
            return f"SUB {args[0]}, {args[1]}, {args[2]}"
        
        elif op == 'MUL':
            return f"MUL {args[0]}, {args[1]}, {args[2]}"
        
        elif op == 'DIV':
            return f"DIV {args[0]}, {args[1]}, {args[2]}"
        
        elif op == 'POW':
            return f"POW {args[0]}, {args[1]}, {args[2]}"

        elif op == 'CMP':
            return f"CMP {args[0]}, {args[1]}, {args[2]}, {args[3]}"
        
        elif op == 'STORE':
            return f"STORE {args[0]}, {args[1]}"
        
        elif op == 'LIST':
            return instr
        
        elif op == 'STRING_OP':
            return instr
        
        elif op == 'JMP_IF_TRUE':
            return f"JMP_IF_TRUE {args[0]}, {args[1]}"
        
        elif op == 'JMP_IF_FALSE':
            return f"JMP_IF_FALSE {args[0]}, {args[1]}"
        
        elif op == 'LABEL':
            return f"LABEL {args[0]}"
        
        elif op == 'CALL':
            return f"CALL {args[0]}, {args[1]}"

        elif op == 'APPEND':
            list1_clean = clean_arg(arg1)
            list2_clean = clean_arg(arg2)
            assembly.append(f"APPEND {dest}, {list1_clean}, {list2_clean}")

        elif op == 'REMOVE':
            list1_clean = clean_arg(arg1)
            list2_clean = clean_arg(arg2)
            assembly.append(f"REMOVE {dest}, {list1_clean}, {list2_clean}")

        elif op == 'SET':
            value_clean = clean_arg(arg1)
            assembly.append(f"SET {dest}, {value_clean}")

        elif op == 'LENGTH':
            list_clean = clean_arg(arg1)
            assembly.append(f"LENGTH {dest}, {list_clean}")      

        elif op == 'DISPLAY':
            assembly.append(f"DISPLAY {dest}")

        elif op == 'PRINT':
            assembly.append(f"PRINT {dest}")

        elif op == 'LOAD':
            assembly.append(f"LOAD {dest}, \"{arg1_clean}\"")

        elif op in ['ADD', 'SUB', 'MUL', 'DIV', 'POW']:
            assembly.append(f"{op} {dest}, {arg1_clean}, {arg2_clean}")

        elif op == 'CREATE':
            list_var, *elements = args
            clean_elements = [clean_arg(e) for e in elements if e is not None]
            if clean_elements:
                assembly.append(f"CREATE {list_var}, {', '.join(clean_elements)}")
            else:
                assembly.append(f"CREATE {list_var}")

        elif op == 'ASSIGN':
            if arg1_clean:
                assembly.append(f"{op} {dest}, {arg1_clean}")
            else:
                assembly.append(f"{op} {dest}")

        elif op == 'BARCHART':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"BARCHART {dest}, {chart_data_clean}")

        elif op == 'COLUMNCHART':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"COLUMNCHART {dest}, {chart_data_clean}")

        elif op == 'LINECHART':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"LINECHART {dest}, {chart_data_clean}")

        elif op == 'PIECHART':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"PIECHART {dest}, {chart_data_clean}")

        elif op == 'HISTOGRAM':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"HISTOGRAM {dest}, {chart_data_clean}")

        elif op == 'SCATTERPLOT':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"SCATTERPLOT {dest}, {chart_data_clean}")

        elif op == 'BUBBLECHART':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"BUBBLECHART {dest}, {chart_data_clean}")

        elif op == 'BOXPLOT':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"BOXPLOT {dest}, {chart_data_clean}")

        elif op == 'HEATMAP':
            chart_data_clean = clean_arg(arg1)
            assembly.append(f"HEATMAP {dest}, {chart_data_clean}")

        elif op == 'FLOWCHART':
            flow_data_clean = clean_arg(arg1)
            assembly.append(f"FLOWCHART {dest}, {flow_data_clean}")

        elif op == 'CIRCLE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"CIRCLE {dest}, {dimensions_clean}")

        elif op == 'SQUARE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"SQUARE {dest}, {dimensions_clean}")

        elif op == 'RECTANGLE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"RECTANGLE {dest}, {dimensions_clean}")

        elif op == 'OVAL':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"OVAL {dest}, {dimensions_clean}")

        elif op == 'TRIANGLE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"TRIANGLE {dest}, {dimensions_clean}")

        elif op == 'RHOMBUS':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"RHOMBUS {dest}, {dimensions_clean}")

        elif op == 'POLYGON':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"POLYGON {dest}, {dimensions_clean}")

        elif op == 'CUBE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"CUBE {dest}, {dimensions_clean}")

        elif op == 'SPHERE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"SPHERE {dest}, {dimensions_clean}")

        elif op == 'CONE':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"CONE {dest}, {dimensions_clean}")

        elif op == 'CYLINDER':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"CYLINDER {dest}, {dimensions_clean}")

        elif op == 'PYRAMID':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"PYRAMID {dest}, {dimensions_clean}")

        elif op == 'CUBOID':
            dimensions_clean = clean_arg(arg1)
            assembly.append(f"CUBOID {dest}, {dimensions_clean}")

        elif op == 'SHOW':
            assembly.append(f"SHOW {dest}")

        else:
            all_args = [clean_arg(a) for a in args if a is not None]
            assembly.append(f"{op} {', '.join(all_args)}")

    return assembly
    
def build_assembly(self):
        """Generates the full assembly-like code from IR instructions."""
        for ir_instruction in self.ir_generator.instructions:
            assembly_line = self.generate(ir_instruction)
            self.assembly.append(assembly_line)
        return "\n".join(self.assembly)


#################################
#### MACHINE CODE STIMULATION ####
#################################

def draw_barchart(values):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(values)), values, color='skyblue')
    plt.title("Bar Chart")
    plt.grid(True, axis='y')
    plt.tight_layout()

def draw_linechart(values):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(values)), values, marker='o', linestyle='-', color='blue')
    plt.title("Line Chart")
    plt.grid(True)
    plt.tight_layout()

def draw_piechart(values):
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=[f"Part {i+1}" for i in range(len(values))], autopct='%1.1f%%')
    plt.title("Pie Chart")
    plt.tight_layout()

def draw_histogram(values):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins='auto', color='orange', edgecolor='black')
    plt.title("Histogram")
    plt.grid(True)
    plt.tight_layout()

def draw_scatterplot(x_values, y_values):
    plt.figure(figsize=(6, 4))
    plt.scatter(x_values, y_values, color='green')
    plt.title("Scatter Plot")
    plt.grid(True)
    plt.tight_layout()

def draw_bubblechart(x_values, y_values, sizes):
    plt.figure(figsize=(6, 4))
    plt.scatter(x_values, y_values, s=[s * 20 for s in sizes], alpha=0.5, c='red', edgecolors='w')
    plt.title("Bubble Chart")
    plt.grid(True)
    plt.tight_layout()

def draw_boxplot(data):
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, vert=True, patch_artist=True)
    plt.grid(True)
    plt.tight_layout()

def draw_heatmap(size, data):
    matrix = np.array(data).reshape((size, size))
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap")
    plt.tight_layout()

def draw_flowchart(data):
    plt.figure(figsize=(6, 4))
    G = nx.DiGraph()
    for i in range(len(data) - 1):
        G.add_edge(data[i], data[i+1])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True, node_size=2000)
    plt.title("Flowchart")
    plt.tight_layout()

def draw_circle(params):
    radius = int(params[0])
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), radius, color='blue', fill=True)
    ax.add_artist(circle)
    ax.set_xlim(-radius-10, radius+10)
    ax.set_ylim(-radius-10, radius+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_square(params):
    side = int(params[0])
    fig, ax = plt.subplots()
    square = plt.Rectangle((-side/2, -side/2), side, side, color='green', fill=True)
    ax.add_artist(square)
    ax.set_xlim(-side-10, side+10)
    ax.set_ylim(-side-10, side+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_rectangle(params):
    width = int(params[0])
    height = int(params[1])
    fig, ax = plt.subplots()
    rectangle = plt.Rectangle((-width/2, -height/2), width, height, color='red', fill=True)
    ax.add_artist(rectangle)
    ax.set_xlim(-width-10, width+10)
    ax.set_ylim(-height-10, height+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_oval(params):
    width = int(params[0])
    height = int(params[1])
    fig, ax = plt.subplots()
    oval = plt.Ellipse((0, 0), width, height, color='orange', fill=True)
    ax.add_artist(oval)
    ax.set_xlim(-width/2-10, width/2+10)
    ax.set_ylim(-height/2-10, height/2+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_triangle(params):
    x1, y1, x2, y2, x3, y3 = map(int, params)
    fig, ax = plt.subplots()
    triangle = plt.Polygon([(x1, y1), (x2, y2), (x3, y3)], color='purple', fill=True)
    ax.add_artist(triangle)
    ax.set_xlim(min(x1, x2, x3)-10, max(x1, x2, x3)+10)
    ax.set_ylim(min(y1, y2, y3)-10, max(y1, y2, y3)+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_rhombus(params):
    x1, y1, x2, y2 = map(int, params)
    fig, ax = plt.subplots()
    rhombus = plt.Polygon([(0, y1), (x1, 0), (0, -y2), (-x2, 0)], color='yellow', fill=True)
    ax.add_artist(rhombus)
    ax.set_xlim(-x1-10, x1+10)
    ax.set_ylim(-y2-10, y2+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_polygon(params):
    points = [tuple(map(int, params[i:i+2])) for i in range(0, len(params), 2)]
    fig, ax = plt.subplots()
    polygon = plt.Polygon(points, color='brown', fill=True)
    ax.add_artist(polygon)
    ax.set_xlim(min([p[0] for p in points])-10, max([p[0] for p in points])+10)
    ax.set_ylim(min([p[1] for p in points])-10, max([p[1] for p in points])+10)
    ax.set_aspect('equal', 'box')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_cube(params):
    side = int(params[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vertices = [
        [0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0],
        [0, 0, side], [side, 0, side], [side, side, side], [0, side, side]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[3], vertices[0], vertices[4], vertices[7]]
    ]
    for face in faces:
        ax.plot_trisurf([v[0] for v in face], [v[1] for v in face], [v[2] for v in face], color='blue')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_sphere(params):
    radius = int(params[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='green')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_cone(params):
    radius = int(params[0])
    height = int(params[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = np.linspace(0, height, 100)
    x = radius * (1 - z / height) * np.cos(np.linspace(0, 2 * np.pi, 100))
    y = radius * (1 - z / height) * np.sin(np.linspace(0, 2 * np.pi, 100))
    ax.plot_trisurf(x, y, z, color='red')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_cylinder(params):
    radius = int(params[0])
    height = int(params[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = np.linspace(0, height, 100)
    x = radius * np.cos(np.linspace(0, 2 * np.pi, 100))
    y = radius * np.sin(np.linspace(0, 2 * np.pi, 100))
    ax.plot_trisurf(x, y, z, color='purple')
    st.pyplot(plt.gcf())
    plt.clf()


def draw_pyramid(x, y, z, size, height):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p0 = [x, y, z]
    p1 = [x + size, y, z]
    p2 = [x + size, y + size, z]
    p3 = [x, y + size, z]

    p4 = [x + size / 2, y + size / 2, z + height]
    
    vertices = [
        [p0, p1, p2, p3], 
        [p0, p1, p4],      
        [p1, p2, p4],
        [p2, p3, p4],
        [p3, p0, p4]
    ]
    
    ax.add_collection3d(Poly3DCollection(vertices, facecolors='m', linewidths=1, edgecolors='r', alpha=.25))
    ax.set_title("PYRAMID")
    
    ax.set_xlim([x - 1, x + size + 1])
    ax.set_ylim([y - 1, y + size + 1])
    ax.set_zlim([z - 1, z + height + 1])
    
    st.pyplot(plt.gcf())
    plt.clf()
   

def resolve_value(val, variables):
    if val in variables:
        val = variables[val]

    try:
        return ast.literal_eval(val)
    except (ValueError, TypeError):
        pass

    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            return ast.literal_eval(val)
        except:
            print(f"[resolve_value] Warning: Failed to parse {val}")
            return []

    if isinstance(val, str) and val.startswith('"') and val.endswith('"'):
        return val[1:-1]

    return val    

def evaluate_condition(lhs, op, rhs, variables):
    lhs = resolve_value(lhs, variables)
    rhs = resolve_value(rhs, variables)
    return {
        '>': lhs > rhs,
        '<': lhs < rhs,
        '>=': lhs >= rhs,
        '<=': lhs <= rhs,
        '==': lhs == rhs,
        '!=': lhs != rhs
    }.get(op, False)

operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.Eq: op.eq,
    ast.NotEq: op.ne
}

def eval_ast(node, variables):
    if isinstance(node, ast.Constant): 
        return node.value
    elif isinstance(node, ast.Name):
        return variables.get(node.id, 0)
    elif isinstance(node, ast.BinOp):
        left = eval_ast(node.left, variables)
        right = eval_ast(node.right, variables)
        return operators[type(node.op)](left, right)
    elif isinstance(node, ast.Compare):
        left = eval_ast(node.left, variables)
        right = eval_ast(node.comparators[0], variables)
        return operators[type(node.ops[0])](left, right)
    else:
        raise TypeError(f"Unsupported AST node: {node}")

def eval_expr(expr_tokens, variables):
    expr_str = ' '.join(expr_tokens)  
    try:
        tree = ast.parse(expr_str, mode='eval')
        return eval_ast(tree.body, variables)
    except Exception as e:
        print(f"Error evaluating expression: {expr_str} -> {e}")
        return None  

def run_and_capture_output(code):
    buffer = io.StringIO()
    sys.stdout = buffer
    execute_code(code)  
    sys.stdout = sys.__stdout__
    return buffer.getvalue()   

def save_plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

image_outputs = []
captured_output = []

def execute_code(code, memory=None):
    memory = memory if memory is not None else {}
    pc = 0  
    machine_code_output = []

    while pc < len(code):
        instr = code[pc]
        if not instr or len(instr) < 1:
            pc += 1
            continue

        cmd = instr[0]
        args = instr[1:]

        if cmd == 'MOV':
            dest, value = args
            memory[dest] = int(memory.get(value, value)) if isinstance(value, str) and value in memory else int(value)

        elif cmd == 'ADD':
            dest, op1, op2 = args
            val1 = int(memory.get(op1, op1))
            val2 = int(memory.get(op2, op2))
            memory[dest] = val1 + val2

        elif cmd == 'SUB':
            dest, op1, op2 = args
            val1 = int(memory.get(op1, op1))
            val2 = int(memory.get(op2, op2))
            memory[dest] = val1 - val2

        elif cmd == 'MUL':
            dest, op1, op2 = args
            val1 = int(memory.get(op1, op1))
            val2 = int(memory.get(op2, op2))
            memory[dest] = val1 * val2

        elif cmd == 'DIV':
            dest, op1, op2 = args
            val1 = int(memory.get(op1, op1))
            val2 = int(memory.get(op2, op2))
            if val2 == 0:
                raise ZeroDivisionError("Division by zero.")
            memory[dest] = val1 // val2

        elif cmd == 'PRINT':
            var = args[0]
            print(memory.get(var, var))

        elif cmd == 'SET':
            var_name = args[0]
            if len(args) == 3 and args[1] == '+':
                val1 = resolve_value(args[0], memory)
                val2 = resolve_value(args[2], memory)
                result = val1 + val2 if not isinstance(val1, str) else str(val1) + str(val2)
                memory[var_name] = result
                machine_code_output.append(f"SET {var_name}, ADD {val1}, {val2}")
            else:
                value = resolve_value(args[1], memory)
                memory[var_name] = value
                machine_code_output.append(f"SET {var_name}, {value}")

        elif cmd == 'CIRCLE':
            args = [resolve_value(x, memory) for x in args[:3]]
            machine_code_output.append(f"DRAW_CIRCLE {args[0]} {args[1]} {args[2]}")
            draw_circle(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'SQUARE':
            args = [resolve_value(x, memory) for x in args[:2]]
            machine_code_output.append(f"DRAW_SQUARE {args[0]} {args[1]}")
            draw_square(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'RECTANGLE':
            args = [resolve_value(x, memory) for x in args[:4]]
            machine_code_output.append(f"DRAW_RECT {args[0]} {args[1]} {args[2]} {args[3]}")
            draw_rectangle(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'TRIANGLE':
            args = [resolve_value(x, memory) for x in args[:6]]
            machine_code_output.append("DRAW_TRIANGLE " + ' '.join(map(str, args)))
            draw_triangle(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'OVAL':
            args = [resolve_value(x, memory) for x in args[:2]]
            machine_code_output.append(f"DRAW_OVAL {args[0]} {args[1]}")
            draw_oval(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'RHOMBUS':
            args = [resolve_value(x, memory) for x in args[:2]]
            machine_code_output.append(f"DRAW_RHOMBUS {args[0]} {args[1]}")
            draw_rhombus(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'POLYGON':
            args = [resolve_value(x, memory) for x in args]
            machine_code_output.append("DRAW_POLYGON " + ' '.join(map(str, args)))
            draw_polygon(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'SPHERE':
            args = [resolve_value(x, memory) for x in args[:4]]
            machine_code_output.append(f"DRAW_SPHERE {args[0]} {args[1]} {args[2]} {args[3]}")
            draw_sphere(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'CUBE':
            args = [resolve_value(x, memory) for x in args[:3]]
            machine_code_output.append(f"DRAW_CUBE {args[0]} {args[1]} {args[2]}")
            draw_cube(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'CYLINDER':
            args = [resolve_value(x, memory) for x in args[:3]]
            machine_code_output.append(f"DRAW_CYLINDER {args[0]} {args[1]} {args[2]}")
            draw_cylinder(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'CONE':
            args = [resolve_value(x, memory) for x in args[:3]]
            machine_code_output.append(f"DRAW_CONE {args[0]} {args[1]} {args[2]}")
            draw_cone(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        elif cmd == 'PYRAMID':
            args = [resolve_value(x, memory) for x in args[:5]]
            machine_code_output.append(f"DRAW_PYRAMID {args[0]} {args[1]} {args[2]} {args[3]} {args[4]}")
            draw_pyramid(*map(float, args))
            image_outputs.append(save_plot_to_base64())

        if cmd == 'BARCHART':
            values =  [eval(v) if isinstance(v, str) else v for v in instr[1:]]
            plt.bar(range(len(values)), values)
            plt.title("Bar Chart")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW BARCHART {' '.join(map(str, values))}")
            image_outputs.append(img_data)

        elif cmd == 'COLUMNCHART':
            values = [eval(v) if isinstance(v, str) else v for v in instr[1:]]
            plt.barh(range(len(values)), values)
            plt.title("Column Chart")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW COLUMNCHART {' '.join(map(str, values))}")
            image_outputs.append(img_data)

        elif cmd == 'LINECHART':
            values = [resolve_value(v, str) for v in instr[1:]]
            plt.plot(values, marker='o')
            plt.title("Line Chart")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW LINECHART {' '.join(map(str, values))}")
            image_outputs.append(img_data)

        elif cmd == 'PIECHART':
            values = [resolve_value(v, str) for v in instr[1:]]
            plt.pie(values, labels=[str(i) for i in range(len(values))], autopct='%1.1f%%')
            plt.title("Pie Chart")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW PIECHART {' '.join(map(str, values))}")
            image_outputs.append(img_data)

        elif cmd == 'HISTOGRAM':
            raw_values = [resolve_value(v, str) for v in instr[1:]]
            values = []
            for v in raw_values:
                if isinstance(v, list):
                    values.extend(v)
                else:
                    values.append(v)
            plt.hist(values, bins='auto', alpha=0.7)
            plt.title("Histogram")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW HISTOGRAM {' '.join(map(str, values))}")
            image_outputs.append(img_data)

        elif cmd == 'SCATTERPLOT':
            x_values = [resolve_value(v, str) for v in instr[1::2]]
            y_values = [resolve_value(v, str) for v in instr[2::2]]
            plt.scatter(x_values, y_values)
            plt.title("Scatter Plot")
            points = ' '.join(f"{x} {y}" for x, y in zip(x_values, y_values))
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW SCATTERPLOT {points}")
            image_outputs.append(img_data)

        elif cmd == 'BUBBLECHART':
            x_vals = [resolve_value(v, str) for v in instr[1::3]]
            y_vals = [resolve_value(v, str) for v in instr[2::3]]
            sizes = [resolve_value(v, str) for v in instr[3::3]]
            plt.scatter(x_vals, y_vals, s=[s*10 for s in sizes], alpha=0.5)
            plt.title("Bubble Chart")
            points = ' '.join(f"{x} {y} {s}" for x, y, s in zip(x_vals, y_vals, sizes))
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW BUBBLECHART {points}")
            image_outputs.append(img_data)

        elif cmd == 'BOXPLOT':
            data = [resolve_value(v, str) for v in instr[1:]]
            plt.boxplot(data)
            plt.title("Box Plot")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW BOXPLOT {' '.join(map(str, data))}")
            image_outputs.append(img_data)

        elif cmd == 'HEATMAP':
            size = int(resolve_value(instr[1], str))
            flat_data = [resolve_value(v, str) for v in instr[2:]]
            matrix = [flat_data[i*size:(i+1)*size] for i in range(size)]
            plt.imshow(matrix, cmap='hot', interpolation='nearest')
            plt.title("Heatmap")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW HEATMAP {size} {' '.join(map(str, flat_data))}")
            image_outputs.append(img_data)

        elif cmd == 'FLOWCHART':
            list_name = instr[1]
            flow_data = list.get(list_name, [])
            plt.plot(flow_data)
            plt.title("Flowchart (Simulated)")
            img_data = save_plot_to_base64()
            machine_code_output.append(f"DRAW FLOWCHART {list_name}")
            image_outputs.append(img_data)

        elif cmd == 'CREATE' and instr[1] == 'LIST':
            list_name = instr[2]
            list[list_name] = []
            machine_code_output.append(f"LIST CREATE {list_name}")

        elif cmd == 'APPEND':
            list_name = instr[1]
            value = resolve_value(instr[2], str)
            list[list_name].append(value)
            machine_code_output.append(f"LIST APPEND {list_name} {value}")

        elif cmd == 'REMOVE':
            list_name = instr[1]
            value = resolve_value(instr[2], str)
            if value in list[list_name]:
                list[list_name].remove(value)
                machine_code_output.append(f"LIST REMOVE {list_name} {value}")

        elif cmd == 'LENGTH':
            list_name = instr[1]
            length = len(list[list_name])
            machine_code_output.append(f"LIST LENGTH {list_name}  // {length}")

        elif cmd == 'DISPLAY':
            list_name = instr[1]
            if list_name in list:
                machine_code_output.append(f"LIST DISPLAY {list_name}  // {list[list_name]}")
            else:
                machine_code_output.append(f"LIST DISPLAY {list_name}  // List not found")

        elif cmd == 'IF':
            var1 = resolve_value(instr[1], str)
            op = instr[2]
            var2 = resolve_value(instr[3], str)
            machine_code_output.append(f"IF {var1} {op} {var2}")
            condition = eval(f"{var1} {op} {var2}")
            i += 1
            if condition:
                while i < len(code) and code[i][0] not in ('ELSE', 'ENDIF'):
                    machine_code_output.extend(execute_code([code[i]], str, list))
                    i += 1
            else:
                while i < len(code) and code[i][0] != 'ELSE':
                    i += 1
                if i < len(code) and code[i][0] == 'ELSE':
                    i += 1
                    while i < len(code) and code[i][0] != 'ENDIF':
                        machine_code_output.extend(execute_code([code[i]], str, list))
                        i += 1
            machine_code_output.append("ENDIF")

        elif cmd == 'FOR':
            var = instr[1]
            start = resolve_value(instr[3], str)
            end = resolve_value(instr[5], str)
            machine_code_output.append(f"FOR {var} FROM {start} TO {end}")
            loop_body = []
            j = i + 1
            while code[j][0] != 'ENDFOR':
                loop_body.append(code[j])
                j += 1
            for val in range(start, end + 1):
                str[var] = val
                machine_code_output.extend(execute_code(loop_body, str, list))
            machine_code_output.append("ENDFOR")
            i = j

        elif cmd == 'WHILE':
            cond = ' '.join(instr[1:])
            machine_code_output.append(f"WHILE {cond}")
            loop_body = []
            j = i + 1
            while j < len(code) and code[j][0] != 'ENDWHILE':
                loop_body.append(code[j])
                j += 1
            while eval(cond, {}, str):
                machine_code_output.extend(execute_code(loop_body, str, list))
            machine_code_output.append("ENDWHILE")
            i = j

        else:
            print(f"Unknown command: {cmd}")
            machine_code_output.append(f"UNKNOWN_CMD {instr}")

        pc += 1

    return machine_code_output, image_outputs

        
#################
### Pipeline#####
#################
def compile_dazl(code, filename):
    lexer = Lexer(filename, code)
    tokens, error = lexer.make_tokens()
    if error:
        print(error.as_string())
        return

    parser = Parser(tokens)
    parse_result = parser.parse()
    if parse_result.error:
        print(parse_result.error.as_string())
        return
    ast = parse_result.node

    semantic_analyzer = SemanticAnalyzer()
    try:
        sem_ast = semantic_analyzer.analyze(ast)
    except SemanticError as e:
        print(f"Semantic Error: {e.message}")
        return

    ir_generator = IRGenerator()
    if isinstance(ast, list):
      for stmt in ast:
        ir_generator.generate(stmt)
    else:
      ir_generator.generate(ast)

    ir = ir_generator.instructions  

    optimizer = Optimizer(ir)  
    optimizer.optimize_ir() 

    assembly = generate_assembly(optimizer)

    machine_code = execute_code(assembly)

    return {
    "text_output": captured_output,
    "images": image_outputs
}

def compile_from_code_string(code_str):
    buffer = StringIO()
    import sys
    original_stdout = sys.stdout
    sys.stdout = buffer  # Redirect print output

    try:
        lines = code_str.strip().splitlines()
        tokens = tokenize(lines)
        ast_rep = parse(tokens)
        analyzed = semantic_analysis(ast_rep)
        ir = generate_ir(analyzed)
        assembly = generate_assembly(ir)
        execute_code(assembly)
    finally:
        sys.stdout = original_stdout  # Reset print output

    return buffer.getvalue()

# Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dazl_compiler.py <filename.dazl>")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, 'r') as f:
        code = f.read()

    compile_dazl(code, filename)  
