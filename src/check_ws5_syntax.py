#!/usr/bin/env python3
"""Quick syntax check + test for WS5."""
import ast
import sys

with open("src/ws5_gradio.py") as f:
    src = f.read()

try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)

# Count functions and classes
tree = ast.parse(src)
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
print(f"Functions: {funcs}")
print(f"Classes: {classes}")
print(f"Lines: {len(src.splitlines())}")
