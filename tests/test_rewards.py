"""Unit tests for reward functions."""

from src.training.rewards import (
    combined_reward,
    extract_code_block,
    json_reward,
    python_reward,
    reasoning_reward,
)


class TestExtractCodeBlock:
    def test_json_fenced(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert extract_code_block(text, "json") == '{"key": "value"}'

    def test_python_fenced(self):
        text = "Here is code:\n```python\ndef foo():\n    return 42\n```"
        assert extract_code_block(text, "python") == "def foo():\n    return 42"

    def test_generic_fence_fallback(self):
        text = "```\n{}\n```"
        assert extract_code_block(text, "json") == "{}"

    def test_no_fence_json_raw(self):
        text = '{"name": "test"}'
        assert extract_code_block(text, "json") == '{"name": "test"}'

    def test_no_fence_python_raw(self):
        text = "def hello():\n    pass"
        assert extract_code_block(text, "python") == "def hello():\n    pass"

    def test_no_match(self):
        text = "Just some plain text with no code."
        assert extract_code_block(text, "json") is None


class TestJsonReward:
    def test_valid_json(self):
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        assert json_reward(text) == 1.0

    def test_valid_array(self):
        text = "```json\n[1, 2, 3]\n```"
        assert json_reward(text) == 1.0

    def test_invalid_json(self):
        text = '```json\n{"name": "Alice", age: 30}\n```'
        assert json_reward(text) == 0.0

    def test_no_code_block(self):
        text = "No JSON here."
        assert json_reward(text) == 0.0

    def test_partial_credit(self):
        # Missing closing brace — error near end
        text = '```json\n{"name": "Alice", "age": 30\n```'
        reward = json_reward(text, partial_credit=True)
        assert reward in (0.0, 0.5)


class TestPythonReward:
    def test_valid_function(self):
        text = (
            "```python\n"
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "```"
        )
        assert python_reward(text) == 1.0

    def test_valid_class(self):
        text = "```python\nclass Stack:\n    def __init__(self):\n        self.items = []\n```"
        assert python_reward(text) == 1.0

    def test_invalid_syntax(self):
        text = "```python\ndef broken(\n    return\n```"
        assert python_reward(text) == 0.0

    def test_no_code_block(self):
        text = "No Python here."
        assert python_reward(text) == 0.0


class TestReasoningReward:
    def test_with_reasoning(self):
        text = (
            "<think>Let me think about this problem carefully"
            " and plan the solution step by step.</think>\n"
            "```json\n{}\n```"
        )
        assert reasoning_reward(text) == 0.2

    def test_without_reasoning(self):
        text = '```json\n{"key": "val"}\n```'
        assert reasoning_reward(text) == 0.0

    def test_short_reasoning(self):
        text = "<think>ok</think>\n```json\n{}\n```"
        assert reasoning_reward(text) == 0.0  # less than 20 chars


class TestCombinedReward:
    def test_json_valid(self):
        text = '```json\n{"x": 1}\n```'
        assert combined_reward(text, "json") == 1.0

    def test_python_valid(self):
        text = "```python\nx = 1\n```"
        assert combined_reward(text, "python") == 1.0

    def test_with_reasoning_bonus(self):
        text = (
            "<think>I need to generate a valid JSON object"
            " with the required keys and types.</think>\n"
            '```json\n{"a": 1}\n```'
        )
        reward = combined_reward(text, "json", reasoning_bonus=0.2)
        assert reward == 1.2  # 1.0 (parse) + 0.2 (reasoning)

    def test_invalid_task_type(self):
        try:
            combined_reward("text", "unknown")
            assert False, "Should raise ValueError"
        except ValueError:
            pass
