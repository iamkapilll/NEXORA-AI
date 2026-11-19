"""Safe arithmetic calculator tool."""
import ast
import math
import operator
from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 256
MAX_EXPRESSION_NODES = 64
MAX_POWER_EXPONENT = 100
MAX_POWER_RESULT_BITS = 4096


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should use arithmetic or an approved built-in or math function.
    Expressions are limited in size and complexity.
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
    allowed_math_functions = {
        name: getattr(math, name)
        for name in (
            "acos", "asin", "atan", "ceil", "cos", "degrees", "exp",
            "fabs", "floor", "log", "log10", "log2", "radians", "sin",
            "sqrt", "tan",
        )
    }

    if len(expression) > MAX_EXPRESSION_LENGTH:
        return "Calculation error: Expression is too long."

    def evaluate(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and (not isinstance(node.value, float) or math.isfinite(node.value))
        ):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError("Only numeric arithmetic is allowed.")
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_POWER_EXPONENT:
                    raise ValueError("Power exponent is too large.")
                if left:
                    estimated_bits = math.log2(abs(left)) * right
                    if estimated_bits > MAX_POWER_RESULT_BITS:
                        raise ValueError("Power result is too large.")
            return binary_operators[type(node.op)](left, right)

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
            ):
                function = allowed_math_functions.get(node.func.attr)
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
        if sum(1 for _ in ast.walk(expression_tree)) > MAX_EXPRESSION_NODES:
            raise ValueError("Expression is too complex.")
        return str(evaluate(expression_tree.body))

    except Exception as e:
        return f"Calculation error: {str(e)}"
