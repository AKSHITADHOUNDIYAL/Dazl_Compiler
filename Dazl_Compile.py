import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
from matplotlib.patches import Rectangle, Polygon , Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.dates as mdates
import mplfinance as mpf
from matplotlib.patches import Ellipse
import matplotlib.dates as mdates
from itertools import product, combinations
import os
import ast
import json
import datetime
import operator as op

def tokenize(lines):
    tokens = []
    for line in lines :
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = re.findall(r'"[^"]*"|\w+|\+|\-|\*|\/|\^|==|!=|<=|>=|<|>|=|\(|\)|\[|\]|,|THEN|TO', line)
        tokens.append(parts)
    return tokens

def parse(tokens):
    return tokens

def semantic_analysis(ast):
    return ast

def generate_ir(ast):
    return ast

def generate_assembly(ir):
    return ir

def resolve_value(val, variables):
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val in variables:
            return variables[val]
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
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

def eval_expr(expr, variables):
    try:
        node = ast.parse(expr, mode='eval').body
        return eval_ast(node, variables)
    except Exception as e:
        print(f"Error evaluating expression: {expr}", e)
        return None
    
def eval_binary_op(op_node, left, right):
    if isinstance(op_node, ast.Add): return left + right
    if isinstance(op_node, ast.Sub): return left - right
    if isinstance(op_node, ast.Mult): return left * right
    if isinstance(op_node, ast.Div): return left / right
    if isinstance(op_node, ast.Mod): return left % right

def eval_compare_op(op_node, left, right):
    if isinstance(op_node, ast.Lt): return left < right
    if isinstance(op_node, ast.Gt): return left > right
    if isinstance(op_node, ast.LtE): return left <= right
    if isinstance(op_node, ast.GtE): return left >= right
    if isinstance(op_node, ast.Eq): return left == right
    if isinstance(op_node, ast.NotEq): return left != right    

def eval_ast(node, variables):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return variables.get(node.id, node.id)
    elif isinstance(node, ast.BinOp):
        left = eval_ast(node.left, variables)
        right = eval_ast(node.right, variables)
        return eval_binary_op(node.op, left, right)
    elif isinstance(node, ast.Compare):
        left = eval_ast(node.left, variables)
        right = eval_ast(node.comparators[0], variables)
        return eval_compare_op(node.ops[0], left, right)
    elif isinstance(node, ast.BoolOp):
        values = [eval_ast(v, variables) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    return None

def eval_expr(expr_tokens, variables):
    expr_str = ' '.join(expr_tokens)  
    try:
        tree = ast.parse(expr_str, mode='eval')
        return eval_ast(tree.body, variables)
    except Exception as e:
        print(f"Error evaluating expression: {expr_str} -> {e}")
        return None

combined_shapes = []
shape_buffer = []
variables = {}

def render_flowchart(flow_data):
    try:
        num_steps = len(flow_data)
        box_height = 1
        box_spacing = 1.5
        fig_height = num_steps * (box_height + box_spacing)

        fig, ax = plt.subplots(figsize=(6, fig_height))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, fig_height)
        ax.axis('off') 

        for idx, step in enumerate(flow_data):
            y = fig_height - (idx + 1) * (box_height + box_spacing / 2)

            box = FancyBboxPatch((3, y), 4, box_height,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightblue')
            ax.add_patch(box)
            ax.text(5, y + box_height / 2, step, ha='center', va='center', fontsize=10)

            if idx < num_steps - 1:
                arrow_start_y = y - 0.1
                arrow_end_y = y - box_spacing
                ax.annotate('', xy=(5, y - 0.2), xytext=(5, y - 0.8),
                        arrowprops=dict(arrowstyle="<-", lw=1.5, shrinkA=0, shrinkB=0))

        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

    except Exception as e:
        print("Error drawing flowchart:", e)

def validate_flowchart_data(flow_data):
    if not isinstance(flow_data, list):
        raise ValueError("Flowchart data must be a list of steps.")
    
    if len(flow_data) < 2:
        raise ValueError("Flowchart must have at least 2 steps (start and end).")

    return flow_data


def draw_square(x, y, side, ax):
    square = Rectangle((x, y), side, side, fill=False, edgecolor='blue')
    ax.add_patch(square)

def draw_triangle(x1, y1, x2, y2, x3, y3, ax):
    triangle = Polygon([(x1, y1), (x2, y2), (x3, y3)], closed=True, fill=False, edgecolor='green')
    ax.add_patch(triangle)

def draw_circle(x, y, r, ax):
    circle = Circle((x, y), r, fill=False, edgecolor='orange')
    ax.add_patch(circle)

def draw_rectangle(x, y, width, height, ax):
    rect = Rectangle((x, y), width, height, fill=False, edgecolor='red')
    ax.add_patch(rect)

def draw_oval(x, y, width, height, ax):
    oval = Ellipse((x, y), width, height, fill=False, edgecolor='purple')
    ax.add_patch(oval)

def draw_rhombus(x, y, d1, d2, ax):
    half_d1, half_d2 = d1 / 2, d2 / 2
    points = [(x, y + half_d2), (x + half_d1, y), (x, y - half_d2), (x - half_d1, y)]
    rhombus = Polygon(points, closed=True, fill=False, edgecolor='cyan')
    ax.add_patch(rhombus)

def draw_polygon(points, ax):
    polygon = Polygon(points, closed=True, fill=False, edgecolor='brown')
    ax.add_patch(polygon)

def draw_cube(x, y, z, size, ax3d):
    r = [0, size]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == size:
            ax3d.plot3D(*zip(s + np.array([x, y, z]), e + np.array([x, y, z])), color="blue")

def draw_cuboid(x, y, z, l, w, h, ax3d):
    points = np.array([[0, 0, 0], [l, 0, 0], [l, w, 0], [0, w, 0],
                       [0, 0, h], [l, 0, h], [l, w, h], [0, w, h]])
    points += np.array([x, y, z])
    edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    for i,j in edges:
        ax3d.plot3D(*zip(points[i], points[j]), color="black")

def draw_cylinder(x, y, z, r, h, ax3d):
    theta = np.linspace(0, 2*np.pi, 30)
    z_vals = np.linspace(z, z+h, 30)
    theta, z_vals = np.meshgrid(theta, z_vals)
    X = x + r * np.cos(theta)
    Y = y + r * np.sin(theta)
    Z = z_vals
    ax3d.plot_surface(X, Y, Z, color='cyan', alpha=0.6)

def draw_sphere(x, y, z, r, ax3d):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = x + r * np.cos(u) * np.sin(v)
    Y = y + r * np.sin(u) * np.sin(v)
    Z = z + r * np.cos(v)
    ax3d.plot_surface(X, Y, Z, color='magenta', alpha=0.6)

def draw_cone(x, y, z, r, h, ax3d):
    theta = np.linspace(0, 2 * np.pi, 30)
    R = np.linspace(0, r, 30)
    T, R = np.meshgrid(theta, R)
    X = x + R * np.cos(T)
    Y = y + R * np.sin(T)
    Z = z + h * (1 - R / r)
    ax3d.plot_surface(X, Y, Z, color='green', alpha=0.6)

def draw_pyramid(x, y, z, size, height, ax3d):
    p0 = [x, y, z]
    p1 = [x + size, y, z]
    p2 = [x + size, y + size, z]
    p3 = [x, y + size, z]
    p4 = [x + size/2, y + size/2, z + height]
    faces = [
        [p0, p1, p2, p3],
        [p0, p1, p4],
        [p1, p2, p4],
        [p2, p3, p4],
        [p3, p0, p4],
    ]
    for face in faces:
        face = np.array(face)
        ax3d.plot_trisurf(face[:, 0], face[:, 1], face[:, 2], color='orange', alpha=0.6)

def parse_flowchart_input(arg_str): 
    global variables  # ensure we access the global variables dictionary
    flow_data = None
    try:
        arg_str = arg_str.strip()
        if not arg_str:
            print("FLOWCHART input is empty.")
            return None

        if arg_str in variables:
            flow_data = variables[arg_str]
            if isinstance(flow_data, str): 
                flow_data = ast.literal_eval(flow_data)
        else:
            try:
                flow_data = ast.literal_eval(arg_str)
            except Exception:
                parts = [part.strip().strip('"').strip("'") for part in arg_str.split(',')]
                flow_data = parts
    except Exception as e:
        print(f"Error interpreting FLOWCHART input: {e}")
    
    return flow_data



def execute_code(code, parent_vars=None, parent_lists=None):
    if parent_vars is None:
        variables = {}
    else:
        variables = parent_vars  

    if parent_lists is None:
        lists = {}
    else:
        lists = parent_lists 

    if parent_vars is None:
        parent_vars = {}
    if parent_lists is None:
        parent_lists = {}

    is_2d = False
    is_3d = False

    for line in code:
        line_str = ' '.join(map(str, line)).strip().upper()
        if any(shape in line_str for shape in ["CIRCLE", "RECTANGLE", "SQUARE", "TRIANGLE", "OVAL", "RHOMBUS", "POLYGON"]):
            is_2d = True
        if any(shape in line_str for shape in ["CUBE", "CUBOID", "CYLINDER", "SPHERE", "CONE", "PYRAMID"]):
            is_3d = True

    fig = plt.figure(figsize=(8, 6))
    ax2d = None
    ax3d = None

    if is_2d and not is_3d:
        ax2d = fig.add_subplot(111)
    elif is_3d and not is_2d:
        ax3d = fig.add_subplot(111, projection='3d')
    elif is_2d and is_3d:
        fig.clf()
        ax2d = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

    def flush_shape_buffer():
        nonlocal ax2d, ax3d
        if not combined_shapes:
            return

        all_x, all_y = [], []
        has_2d = False
        has_3d = False

        if ax2d:
            ax2d.set_aspect('equal')
        if ax3d:
            ax3d.set_axis_off()

        for shape in combined_shapes:
            shape_type, params = shape

            # 2D Shapes
            if shape_type == 'SQUARE':
                x, y, side = params
                draw_square(x, y, side, ax2d)
                all_x.extend([x, x + side])
                all_y.extend([y, y + side])
                has_2d = True

            elif shape_type == 'TRIANGLE':
                x1, y1, x2, y2, x3, y3 = params
                draw_triangle(x1, y1, x2, y2, x3, y3, ax2d)
                all_x.extend([x1, x2, x3])
                all_y.extend([y1, y2, y3])
                has_2d = True

            elif shape_type == 'CIRCLE':
                x, y, r = params
                circle = plt.Circle((x, y), r, color='blue', fill=False)
                ax2d.add_patch(circle)
                all_x.extend([x - r, x + r])
                all_y.extend([y - r, y + r])
                has_2d = True

            elif shape_type == 'RECTANGLE':
                x, y, w, h = params
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red')
                ax2d.add_patch(rect)
                all_x.extend([x, x + w])
                all_y.extend([y, y + h])
                has_2d = True

            elif shape_type == 'OVAL':
                x, y, w, h = params
                ellipse = Ellipse((x, y), w, h, fill=False, edgecolor='blue')
                ax2d.add_patch(ellipse)
                all_x.extend([x - w / 2, x + w / 2])
                all_y.extend([y - h / 2, y + h / 2])
                has_2d = True

            elif shape_type == 'RHOMBUS':
                x, y, d1, d2 = params
                half_d1, half_d2 = d1 / 2, d2 / 2
                points = [
                    (x, y + half_d2),
                    (x + half_d1, y),
                    (x, y - half_d2),
                    (x - half_d1, y)
                ]
                rhombus = Polygon(points, closed=True, edgecolor='blue', fill=False)
                ax2d.add_patch(rhombus)
                all_x.extend([x - half_d1, x + half_d1])
                all_y.extend([y - half_d2, y + half_d2])
                has_2d = True

            elif shape_type == 'POLYGON':
                points = params
                polygon = Polygon(points, closed=True, edgecolor='blue', fill=False)
                ax2d.add_patch(polygon)
                all_x.extend([p[0] for p in points])
                all_y.extend([p[1] for p in points])
                has_2d = True

            # 3D Shapes
            elif shape_type == 'CUBE':
                x, y, z, size = params
                r = [0, size]
                for s, e in combinations(np.array(list(product(r, r, r))), 2):
                    if np.sum(np.abs(s - e)) == size:
                        ax3d.plot3D(*zip(s + np.array([x, y, z]), e + np.array([x, y, z])), color="b")
                has_3d = True

            elif shape_type == 'SPHERE':
                x, y, z, r = params
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                xs = x + r * np.cos(u) * np.sin(v)
                ys = y + r * np.sin(u) * np.sin(v)
                zs = z + r * np.cos(v)
                ax3d.plot_surface(xs, ys, zs, color='r')
                has_3d = True

            elif shape_type == 'CONE':
                x, y, z, r, h = params
                theta = np.linspace(0, 2 * np.pi, 30)
                R = np.linspace(0, r, 30)
                T, R = np.meshgrid(theta, R)
                X = x + R * np.cos(T)
                Y = y + R * np.sin(T)
                Z = z + (h * (1 - R / r))
                ax3d.plot_surface(X, Y, Z, color='g')
                has_3d = True

            elif shape_type == 'CYLINDER':
                x, y, z, r, h = params
                theta = np.linspace(0, 2 * np.pi, 30)
                z_vals = np.linspace(z, z + h, 30)
                theta, z_vals = np.meshgrid(theta, z_vals)
                X = x + r * np.cos(theta)
                Y = y + r * np.sin(theta)
                Z = z_vals
                ax3d.plot_surface(X, Y, Z, color='c')
                has_3d = True

            elif shape_type == 'PYRAMID':
                x, y, z, size, height = params
                p0 = [x, y, z]
                p1 = [x + size, y, z]
                p2 = [x + size, y + size, z]
                p3 = [x, y + size, z]
                p4 = [x + size / 2, y + size / 2, z + height]
                faces = [
                    [p0, p1, p2, p3],
                    [p0, p1, p4],
                    [p1, p2, p4],
                    [p2, p3, p4],
                    [p3, p0, p4]
                ]
                for face in faces:
                    x_vals = [v[0] for v in face]
                    y_vals = [v[1] for v in face]
                    z_vals = [v[2] for v in face]
                    ax3d.plot_trisurf(x_vals, y_vals, z_vals, color='orange', alpha=0.5)
                has_3d = True

        if has_2d:
            if all_x and all_y:
                pad = 10
                ax2d.set_xlim(min(all_x) - pad, max(all_x) + pad)
                ax2d.set_ylim(min(all_y) - pad, max(all_y) + pad)
                ax2d.axis('off')
                plt.show()

        if has_3d:
            ax3d.set_box_aspect([1, 1, 1])
            ax3d.set_axis_off()

        shape_names = [shape[0] for shape in combined_shapes]
        fig.suptitle("Shapes: " + ", ".join(shape_names), fontsize=14)

        plt.tight_layout()
        plt.show()

        shape_buffer.extend(combined_shapes)
        combined_shapes.clear()

    i = 0
    while i < len(code):
        tokens = code[i]
        if not tokens:
            i += 1
            continue

        cmd = tokens[0]

        if cmd == 'PRINT':
            output = []
            i_token = 1
            while i_token < len(tokens):
                token = tokens[i_token]
                if token == '+':
                    prev = output.pop()
                    next_val = resolve_value(tokens[i_token + 1], variables)
                    output.append(str(prev) + str(next_val))
                    i_token += 2
                else:
                    val = resolve_value(token, variables)
                    output.append(str(val))
                    i_token += 1
            print(''.join(output))

        elif cmd == 'SET':
            var_name = tokens[1]
            value_expr = ' '.join(tokens[3:]).strip()  

            if value_expr.startswith('[') or value_expr.startswith('{'):
                collected_lines = [value_expr]
                while True:
                    full_expr = '\n'.join(collected_lines).strip()
                    if full_expr.endswith(']') or full_expr.endswith('}'):
                        break
                    i += 1
                    if i >= len(code):
                        break
                    next_line = str(code[i]).strip()
                    collected_lines.append(next_line)
                try:
                    value = ast.literal_eval(full_expr)
                    variables[var_name] = value
                except Exception as e:
                    print(f"Error evaluating expression for {var_name}: {e}")

            elif len(tokens) == 5 and tokens[2] == '=' and tokens[4] in ['+', '-', '*', '/', '^']:
                print(f"Invalid arithmetic operation in SET: {' '.join(tokens)}")

            elif len(tokens) == 6 and tokens[2] == '=' and tokens[4] in ['+', '-', '*', '/', '^']:
                val1 = resolve_value(tokens[3], variables)
                val2 = resolve_value(tokens[5], variables)
                operator = tokens[4]

                try:
                    if operator == '+':
                        value = val1 + val2
                    elif operator == '-':
                        value = val1 - val2
                    elif operator == '*':
                        value = val1 * val2
                    elif operator == '/':
                        value = val1 / val2
                    elif operator == '^':
                        value = val1 ** val2
                except TypeError as e:
                    print(f"TypeError during operation: {e}")
                    value = str(val1) + str(val2)

            else:
                value = resolve_value(value_expr, variables)
                variables[var_name] = value   

         
        elif cmd == 'SHOW':
            flush_shape_buffer()

        elif cmd == 'CIRCLE':
            cleaned_values = [value.strip('()') for value in tokens[1:4]]
            try:
                x, y, r = map(lambda v: resolve_value(v, variables), cleaned_values)
                x, y, r = float(x), float(y), float(r)
                combined_shapes.append(('CIRCLE', (x, y, r)))
            except ValueError as e:
                print(f"Error: Invalid values for circle: {e}")

        elif cmd == 'SQUARE':
            cleaned_values = [value.strip('()') for value in tokens[1:4]]
            try:
                x, y, side = map(lambda v: resolve_value(v, variables), cleaned_values)
                x, y, side = float(x), float(y), float(side)
                combined_shapes.append(('SQUARE', (x, y, side)))
            except ValueError as e:
                print(f"Error: Invalid values for square: {e}")

        elif cmd == 'RECTANGLE':
            cleaned_values = [value.strip('()') for value in tokens[1:5]]
            try:
                x, y, width, height = map(lambda v: resolve_value(v, variables), cleaned_values)
                x, y, width, height = float(x), float(y), float(width), float(height)
                combined_shapes.append(('RECTANGLE', (x, y, width, height)))
            except ValueError as e:
                print(f"Error: Invalid values for rectangle: {e}")  

        elif cmd == 'OVAL':
            cleaned_values = [value.strip('()') for value in tokens[1:5]]
            try:
                x, y, width, height = map(lambda v: resolve_value(v, variables), cleaned_values)
                x, y, width, height = float(x), float(y), float(width), float(height)
                combined_shapes.append(('OVAL', (x, y, width, height)))
            except ValueError as e:
                print(f"Error: Invalid values for oval: {e}")   

        elif cmd == 'TRIANGLE':
            cleaned_values = [value.strip('()') for value in tokens[1:7]]
            try:
                coords = list(map(lambda v: float(resolve_value(v, variables)), cleaned_values))
                if len(coords) != 6:
                    raise ValueError("TRIANGLE requires 3 coordinate pairs (x1 y1 x2 y2 x3 y3)")
                x1, y1, x2, y2, x3, y3 = coords
                combined_shapes.append(('TRIANGLE', (x1, y1, x2, y2, x3, y3)))
            except ValueError as e:
                print(f"Error: Invalid values for triangle: {e}")     
                     

        elif cmd == 'RHOMBUS':
            cleaned_values = [value.strip('()') for value in tokens[1:4]]
            try:
                x, y, d1, d2 = map(lambda v: resolve_value(v, variables), cleaned_values + [tokens[4]])
                x, y, d1, d2 = float(x), float(y), float(d1), float(d2)
                combined_shapes.append(('RHOMBUS', (x, y, d1, d2)))
            except ValueError as e:
                print(f"Error: Invalid values for rhombus: {e}")
            except IndexError:
                print("Error: RHOMBUS requires 4 values — x, y, d1, d2.")    
            
        elif cmd == 'POLYGON':
            try:
                values = list(map(lambda v: float(resolve_value(v, variables)), tokens[1:]))

                if len(values) % 2 != 0:
                    raise ValueError("Polygon points should be in x, y pairs.")
                points = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
                combined_shapes.append(('POLYGON', points))
            except ValueError as e:
                print(f"Error: Invalid values for polygon: {e}")

        elif cmd == 'CUBE':
            x, y, z, size = map(lambda v: resolve_value(v, variables), tokens[1:5])
            combined_shapes.append(('CUBE', (x, y, z, size)))

        elif cmd == 'CUBOID':
            x, y, z, width, height, depth = map(lambda v: resolve_value(v, variables), tokens[1:7])
            combined_shapes.append(('CUBOID', (x, y, z, size)))    

        elif cmd == 'SPHERE':
            x, y, z, r = map(lambda v: resolve_value(v, variables), tokens[1:5])
            combined_shapes.append(('SPHERE', (x, y, z, r)))

        elif cmd == 'CONE':
            x, y, z, r, h = map(lambda v: resolve_value(v, variables), tokens[1:6])
            combined_shapes.append(('CONE', (x, y, z, r, h)))

        elif cmd == 'CYLINDER':
            x, y, z, r, h = map(lambda v: resolve_value(v, variables), tokens[1:6])
            combined_shapes.append(('CYLINDER', (x, y, z, r, h)))

        elif cmd == 'PYRAMID':
            x, y, z, size, height = map(lambda v: resolve_value(v, variables), tokens[1:6])
            combined_shapes.append(('PYRAMID', (x, y, z, size, height)))
  
        elif cmd == 'COLUMNCHART':
            values = [resolve_value(v, variables) for v in tokens[1:]]
            if all(isinstance(v, (int, float)) for v in values):
                plt.bar(range(len(values)), values)
                plt.title("COLUMNCHART")
                plt.show()
                plt.close()
            else:
                print("COLUMNCHART requires numeric values.")

        elif cmd == 'LINECHART':
            values = [resolve_value(v, variables) for v in tokens[1:]]
            if all(isinstance(v, (int, float)) for v in values):
                plt.plot(values)
                plt.title("LINECHART")
                plt.show()
                plt.close()
            else:
                print("LINECHART requires numeric values.")

        elif cmd == 'PIECHART':
            values = [resolve_value(v, variables) for v in tokens[1:]]
            if all(isinstance(v, (int, float)) for v in values):
                plt.pie(values, labels=[str(v) for v in values])
                plt.title("PIECHART")
                plt.show()
                plt.close()
            else:
                print("PIECHART requires numeric values.")

        elif cmd == 'HISTOGRAM':
            raw_values = [resolve_value(v, variables) for v in tokens[1:]]
            values = []
            for v in raw_values:
                if isinstance(v, list):
                    values.extend(v)
                else:
                    values.append(v)
            plt.hist(values, bins='auto', edgecolor='black')
            plt.title("HISTOGRAM")
            plt.show()
            plt.close()

        elif cmd == 'SCATTERPLOT':
            x_values = [resolve_value(v, variables) for v in tokens[1::2]]
            y_values = [resolve_value(v, variables) for v in tokens[2::2]]
            plt.scatter(x_values, y_values)
            plt.title("SCATTERPLOT")
            plt.show()  
            plt.close()

        elif cmd == 'BUBBLECHART':
            x_vals = [resolve_value(v, variables) for v in tokens[1::3]]
            y_vals = [resolve_value(v, variables) for v in tokens[2::3]]
            sizes = [resolve_value(v, variables) for v in tokens[3::3]]
            plt.scatter(x_vals, y_vals, s=sizes)
            plt.title("BUBBLECHART")
            plt.show()
            plt.close()

        elif cmd == 'BOXPLOT':
            data = [resolve_value(v, variables) for v in tokens[1:]]
            plt.boxplot(data)
            plt.title("BOXPLOT")
            plt.show()
            plt.close()

        elif cmd == 'HEATMAP':
            try:
                size_token = tokens[1]
                size_val = resolve_value(size_token, variables)
                size = int(size_val)  
                raw_data = tokens[2:]
                data = []
                for v in raw_data:
                    val = resolve_value(v, variables)
                    if isinstance(val, list):
                        data.extend(val)
                    else:
                        data.append(val)

                if len(data) != size * size:
                    print(f"Error: Data length {len(data)} does not match size*size ({size*size})")
                    i += 1
                    continue

                matrix = np.array(data, dtype=float).reshape((size, size))
                plt.imshow(matrix, cmap='hot', interpolation='nearest')
                plt.title("HEATMAP")
                plt.colorbar()
                plt.show()
                plt.close()
            except Exception as e:
                print(f"Error in HEATMAP: {e}")        

        elif cmd == 'BARCHART':
            values = [resolve_value(v, variables) for v in tokens[1:]]
            if all(isinstance(v, (int, float)) for v in values):
                plt.barh(range(len(values)), values)
                plt.title("BARCHART")
                plt.show()
                plt.close()
            else:
                print("BARCHART requires numeric values.")

        elif cmd == 'FLOWCHART':
            if len(tokens) < 2:
                print("Invalid FLOWCHART syntax. Expected: FLOWCHART <list> or <variable>")
                i += 1
                continue
            arg_str = " ".join(tokens[1:]).strip()
            flow_data = parse_flowchart_input(arg_str)
            if not validate_flowchart_data(flow_data):
                print("Invalid flowchart data format")
                i += 1
                continue
            try:
                render_flowchart(flow_data)
            except Exception as e:
                print("Error rendering flowchart:", e)

        elif cmd == 'CREATE' and len(tokens) > 2 and tokens[1] == 'LIST':
            lists[tokens[2]] = []

        elif cmd == 'APPEND':
            if tokens[1] in lists:
                lists[tokens[1]].append(resolve_value(tokens[2], variables))
            else:
                print(f"List '{tokens[1]}' not found.")

        elif cmd == 'REMOVE':
            if tokens[1] in lists:
                try:
                    lists[tokens[1]].remove(resolve_value(tokens[2], variables))
                except ValueError:
                    print(f"Value not found in list '{tokens[1]}'.")
            else:
                print(f"List '{tokens[1]}' not found.")

        elif cmd == 'LENGTH':
            if tokens[1] in lists:
                print(len(lists[tokens[1]]))
            else:
                print(f"List '{tokens[1]}' not found.")

        elif cmd == 'DISPLAY':
            if tokens[1] in lists:
                print(lists[tokens[1]])
            else:
                print(f"List '{tokens[1]}' not found.")

        elif cmd == 'IF':
            if len(tokens) < 4:
                print("Invalid IF syntax.")
                i += 1
                continue
            if not evaluate_condition(tokens[1], tokens[2], tokens[3], variables):
                nest = 1
                while i < len(code) - 1:
                    i += 1
                    if code[i][0] == 'IF':
                        nest += 1
                    elif code[i][0] == 'ENDIF':
                        nest -= 1
                        if nest == 0:
                            break
                    elif code[i][0] == 'ELSE' and nest == 1:
                        break
            i += 1
            continue

        elif cmd == 'ELSE':
            while i < len(code) - 1:
                i += 1
                if code[i][0] == 'ENDIF':
                    break
            i += 1
            continue

        elif cmd == 'FOR':
            var = tokens[1]
            start = resolve_value(tokens[3], variables)
            end = resolve_value(tokens[5], variables)
            loop_body = []
            j = i + 1
            while code[j][0] != 'ENDFOR':
                loop_body.append(code[j])
                j += 1
            for val in range(start, end + 1):
                variables[var] = val
                execute_code(loop_body, variables, lists)
            i = j

        elif cmd == 'WHILE':
            if len(tokens) < 4:
                print(f"Invalid WHILE condition: {' '.join(tokens)}")
                i += 1
                continue

            condition = tokens[1:4]
            loop_start = i + 1
            nest = 1
            loop_body = []

            while i + 1 < len(code):
                i += 1
                if code[i][0] == 'WHILE':
                    nest += 1
                elif code[i][0] == 'ENDWHILE':
                    nest -= 1
                    if nest == 0:
                        break
                if code[i][0] not in ['WHILE', 'ENDWHILE']:   
                        loop_body.append(code[i])

            max_iterations = 500
            count = 0
            while evaluate_condition(condition[0], condition[1], condition[2], variables):
                if count >= max_iterations:
                   print("Infinite loop detected. Breaking.")
                   break
                execute_code(loop_body, variables, lists)

                count += 1

            i += 1  
            continue

        elif '=' in tokens and len(tokens) >= 3:
            try:
                eq_index = tokens.index('=')
                var_name = tokens[0]
                expr_tokens = tokens[eq_index + 1:]
                value_str = ' '.join(expr_tokens)

                try:
                    value = eval_expr(expr_tokens, variables)
                    if value is None:
                        value = ast.literal_eval(value_str)
                except Exception:
                    value = ast.literal_eval(value_str)

                variables[var_name] = value
            except Exception as e:
                print(f"Error processing assignment: {' '.join(tokens)} → {e}")
            i += 1
            continue
        
        else:
            print(f"Unknown command: {cmd}")

        i += 1

def compile_and_run(filename):
    with open(filename, "r") as f:
        source = f.readlines()     
    tokens = tokenize(source)
    ast = parse(tokens)
    analyzed = semantic_analysis(ast)
    ir = generate_ir(analyzed)
    assembly = generate_assembly(ir)
    execute_code(assembly)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <source_file>")
    else:
        compile_and_run(sys.argv[1])
