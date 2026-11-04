"""Safe arithmetic calculator tool."""
import ast
import math
import operator
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    allowed_functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
            return binary_operators[type(node.op)](
                evaluate(node.left),
                evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operators:
            return unary_operators[type(node.op)](evaluate(node.operand))

        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(element) for element in node.elts]

        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "math" and not node.attr.startswith("_"):
                value = getattr(math, node.attr, None)
                if isinstance(value, (int, float)):
                    return value

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                function = allowed_functions.get(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and not node.func.attr.startswith("_")
            ):
                function = getattr(math, node.func.attr, None)
                if not callable(function):
                    function = None
            else:
                function = None

            if function is not None and all(keyword.arg is not None for keyword in node.keywords):
                return function(
                    *(evaluate(argument) for argument in node.args),
                    **{
                        keyword.arg: evaluate(keyword.value)
                        for keyword in node.keywords
                    }
                )

        raise ValueError("Only arithmetic expressions and approved math functions are allowed.")

    try:
        expression_tree = ast.parse(expression, mode="eval")
        return str(evaluate(expression_tree.body))

    except Exception as e:
        return f"Calculation error: {str(e)}"
