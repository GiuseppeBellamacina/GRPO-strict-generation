"""Unit tests for reward functions."""

from src.training.rewards import (
    build_reward_functions,
    extract_code_block,
    format_reward,
    reasoning_reward,
    repetition_reward,
    schema_reward,
    strictness_reward,
    truncation_reward,
    validity_reward,
)


class TestExtractCodeBlock:
    def test_json_fenced(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert extract_code_block(text, "json") == '{"key": "value"}'

    def test_generic_fence_fallback(self):
        text = "```\n{}\n```"
        assert extract_code_block(text, "json") == "{}"

    def test_no_fence_json_raw(self):
        text = '{"name": "test"}'
        assert extract_code_block(text, "json") == '{"name": "test"}'

    def test_no_match(self):
        text = "Just some plain text with no code."
        assert extract_code_block(text, "json") is None

    def test_think_then_fenced(self):
        text = '<think>Let me reason about this.</think>\n```json\n{"a": 1}\n```'
        assert extract_code_block(text, "json") == '{"a": 1}'

    def test_think_then_bare_json(self):
        text = '<think>Reasoning here.</think>\n{"a": 1}'
        assert extract_code_block(text, "json") == '{"a": 1}'

    def test_think_then_bare_array(self):
        text = "<think>Some thought.</think>\n[1, 2, 3]"
        assert extract_code_block(text, "json") == "[1, 2, 3]"

    def test_think_only_no_json(self):
        text = "<think>Just thinking, no JSON.</think>"
        assert extract_code_block(text, "json") is None


class TestFormatReward:
    def test_json_fenced_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert format_reward(text) == 1.0

    def test_generic_fenced_block(self):
        text = "```\n{}\n```"
        assert format_reward(text) == 0.5

    def test_no_block(self):
        text = "No code here."
        assert format_reward(text) == 0.0


class TestValidityReward:
    def test_valid_json_object(self):
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        assert validity_reward(text) == 1.0

    def test_valid_json_array(self):
        text = "```json\n[1, 2, 3]\n```"
        assert validity_reward(text) == 1.0

    def test_no_code_block(self):
        text = "No JSON here."
        assert validity_reward(text) == 0.0

    def test_error_near_end(self):
        # Missing closing brace — error near the end
        text = '```json\n{"name": "Alice", "age": 30\n```'
        assert validity_reward(text) == 0.70

    def test_code_block_unparseable(self):
        # Completely unparseable content — error at char 0 → "early mistake"
        text = "```json\nhello world\n```"
        assert validity_reward(text) == 0.20


class TestSchemaReward:
    def test_no_code_block(self):
        assert schema_reward("No code.", "Generate 3 items.") == 0.0

    def test_invalid_json(self):
        assert (
            schema_reward("```json\n{bad}\n```", "Generate 3 items.")
            == 0.0
        )

    def test_exact_array_count_match(self):
        text = "```json\n[1, 2, 3]\n```"
        assert (
            schema_reward(text, "Generate an array of 3 items.")
            == 1.0
        )

    def test_exact_array_count_off_by_one(self):
        text = "```json\n[1, 2]\n```"
        score = schema_reward(text, "Generate an array of 3 items.")
        assert score == 0.70

    def test_required_keys_all_present(self):
        text = '```json\n{"id": 1, "name": "x"}\n```'
        instr = 'Object with "id" (integer) and "name" (string).'
        assert schema_reward(text, instr) == 1.0

    def test_required_keys_partial(self):
        text = '```json\n{"id": 1}\n```'
        instr = 'Object with "id" (integer) and "name" (string).'
        assert schema_reward(text, instr) == 0.5

    def test_no_constraints_valid_json(self):
        text = '```json\n{"a": 1}\n```'
        assert schema_reward(text, "Generate a JSON object.") == 1.0

    def test_toplevel_array_required(self):
        text = "```json\n[1, 2]\n```"
        assert (
            schema_reward(text, "Generate a JSON array of numbers.")
            == 1.0
        )

    def test_toplevel_object_required_but_got_array(self):
        text = "```json\n[1, 2]\n```"
        assert (
            schema_reward(text, "Generate a JSON object with keys.")
            == 0.0
        )

    # --- Schema tag tests ---

    def test_schema_tag_keys_strict_toplevel(self):
        """Schema tag: keys must be at top level, not nested."""
        instr = 'Generate JSON.\n[SCHEMA:{"keys":["name","age"],"toplevel":"object"}]'
        good = '```json\n{"name": "x", "age": 30}\n```'
        assert schema_reward(good, instr) == 1.0
        # Keys present but nested → strict check fails
        bad = '```json\n{"data": {"name": "x", "age": 30}}\n```'
        score = schema_reward(bad, instr)
        assert score < 1.0

    def test_schema_tag_wrong_keys(self):
        """Schema tag: wrong key names penalised."""
        instr = 'Generate JSON.\n[SCHEMA:{"keys":["group","items"],"toplevel":"object"}]'
        wrong = '```json\n{"type": "group", "items": [1,2,3]}\n```'
        right = '```json\n{"group": "test", "items": [1,2,3]}\n```'
        assert schema_reward(right, instr) > schema_reward(
            wrong, instr
        )

    def test_schema_tag_count(self):
        """Schema tag: exact count validation."""
        instr = (
            'Generate array.\n[SCHEMA:{"toplevel":"array","count":3}]'
        )
        exact = "```json\n[1, 2, 3]\n```"
        assert schema_reward(exact, instr) == 1.0
        off = "```json\n[1, 2]\n```"
        assert schema_reward(off, instr) < 1.0

    def test_schema_tag_item_keys(self):
        """Schema tag: item_keys check on array of objects."""
        instr = 'Generate.\n[SCHEMA:{"toplevel":"array","count":2,"item_keys":["name","value"]}]'
        good = '```json\n[{"name":"a","value":1},{"name":"b","value":2}]\n```'
        assert schema_reward(good, instr) == 1.0
        bad = '```json\n[{"label":"a","count":1},{"label":"b","count":2}]\n```'
        # toplevel=1.0, count=1.0, item_keys=0.0 → avg ≈ 0.67
        assert schema_reward(bad, instr) < 1.0
        assert schema_reward(good, instr) > schema_reward(bad, instr)

    def test_schema_tag_min_count(self):
        """Schema tag: minimum count."""
        instr = (
            'Generate.\n[SCHEMA:{"toplevel":"object","min_count":3}]'
        )
        enough = '```json\n{"a":1,"b":2,"c":3,"d":4}\n```'
        assert schema_reward(enough, instr) == 1.0
        too_few = '```json\n{"a":1}\n```'
        assert schema_reward(too_few, instr) < 1.0

    def test_schema_tag_depth(self):
        """Schema tag: nesting depth."""
        instr = 'Generate.\n[SCHEMA:{"toplevel":"object","depth":3}]'
        deep = '```json\n{"a":{"b":{"c":1}}}\n```'
        assert schema_reward(deep, instr) == 1.0
        shallow = '```json\n{"a":1}\n```'
        assert schema_reward(shallow, instr) < 1.0

    def test_schema_tag_fallback_when_absent(self):
        """Without schema tag, regex fallback works as before."""
        text = '```json\n{"id": 1, "name": "x"}\n```'
        instr = 'Object with "id" (integer) and "name" (string).'
        assert schema_reward(text, instr) == 1.0

    # --- key_types tests ---

    def test_key_types_all_correct(self):
        """key_types: all values have correct types → 1.0."""
        instr = '[SCHEMA:{"keys":["name","age","active"],"toplevel":"object","key_types":{"name":"string","age":"integer","active":"boolean"}}]'
        text = '```json\n{"name": "Alice", "age": 30, "active": true}\n```'
        assert schema_reward(text, instr) == 1.0

    def test_key_types_wrong_types(self):
        """key_types: wrong value types penalised."""
        instr = '[SCHEMA:{"keys":["name","age"],"toplevel":"object","key_types":{"name":"string","age":"integer"}}]'
        good = '```json\n{"name": "Alice", "age": 30}\n```'
        bad = '```json\n{"name": 42, "age": "thirty"}\n```'
        assert schema_reward(good, instr) > schema_reward(bad, instr)

    def test_key_types_partial_correct(self):
        """key_types: one correct, one wrong → partial score."""
        instr = '[SCHEMA:{"keys":["name","age"],"toplevel":"object","key_types":{"name":"string","age":"integer"}}]'
        text = '```json\n{"name": "Alice", "age": "thirty"}\n```'
        # keys=1.0, key_types=0.5, toplevel=1.0 → avg = 0.833
        score = schema_reward(text, instr)
        assert 0.5 < score < 1.0

    def test_key_types_array_and_object(self):
        """key_types: array and object types validated correctly."""
        instr = '[SCHEMA:{"keys":["tags","meta"],"toplevel":"object","key_types":{"tags":"array","meta":"object"}}]'
        good = '```json\n{"tags": ["a", "b"], "meta": {"x": 1}}\n```'
        bad = '```json\n{"tags": "not-array", "meta": "not-object"}\n```'
        assert schema_reward(good, instr) == 1.0
        assert schema_reward(good, instr) > schema_reward(bad, instr)

    def test_key_types_boolean_not_integer(self):
        """key_types: boolean must not be treated as integer (Python subclass)."""
        instr = '[SCHEMA:{"keys":["flag"],"toplevel":"object","key_types":{"flag":"integer"}}]'
        good = '```json\n{"flag": 42}\n```'
        bad = '```json\n{"flag": true}\n```'
        assert schema_reward(good, instr) > schema_reward(bad, instr)

    # --- item_key_types tests ---

    def test_item_key_types_all_correct(self):
        """item_key_types: correct types in every item → 1.0."""
        instr = '[SCHEMA:{"toplevel":"array","count":2,"item_keys":["name","value"],"item_key_types":{"name":"string","value":"integer"}}]'
        text = '```json\n[{"name":"a","value":1},{"name":"b","value":2}]\n```'
        assert schema_reward(text, instr) == 1.0

    def test_item_key_types_wrong_types(self):
        """item_key_types: wrong types penalised."""
        instr = '[SCHEMA:{"toplevel":"array","count":2,"item_keys":["name","value"],"item_key_types":{"name":"string","value":"integer"}}]'
        good = '```json\n[{"name":"a","value":1},{"name":"b","value":2}]\n```'
        bad = '```json\n[{"name":1,"value":"x"},{"name":2,"value":"y"}]\n```'
        assert schema_reward(good, instr) > schema_reward(bad, instr)

    def test_item_key_types_nested_object(self):
        """item_key_types: object type validated in items."""
        instr = '[SCHEMA:{"toplevel":"array","count":1,"item_keys":["id","config"],"item_key_types":{"config":"object"}}]'
        good = '```json\n[{"id":"a","config":{"k":1}}]\n```'
        bad = '```json\n[{"id":"a","config":"flat-string"}]\n```'
        assert schema_reward(good, instr) > schema_reward(bad, instr)


class TestReasoningReward:
    def test_with_long_reasoning(self):
        content = (
            "Let me think about this problem carefully and plan the solution step by step. "
            * 3
        )
        text = f"<think>{content}</think>\n```json\n{{}}\n```"
        assert reasoning_reward(text) == 1.0

    def test_with_medium_reasoning(self):
        # 100 chars → capped at 1.0 (plateau at 80)
        content = "I need to create a JSON object with three keys: name, age, and active. Let me structure it properly."
        text = f"<think>{content}</think>\n```json\n{{}}\n```"
        assert reasoning_reward(text) == 1.0

    def test_without_reasoning(self):
        text = '```json\n{"key": "val"}\n```'
        assert reasoning_reward(text) == -0.5

    def test_short_reasoning(self):
        text = "<think>ok</think>\n```json\n{}\n```"
        assert reasoning_reward(text) == -0.2

    def test_graduated_signal(self):
        # 50 chars → (50-10)/70 ≈ 0.57
        content = "A" * 50
        text = f"<think>{content}</think>"
        assert reasoning_reward(text) == pytest.approx((50 - 10) / 70)

    def test_cap_at_one(self):
        content = "A" * 500
        text = f"<think>{content}</think>"
        assert reasoning_reward(text) == 1.0


import pytest  # noqa: E402  (import after class defs to keep test grouping)


class TestTruncationReward:
    def test_fenced_complete(self):
        text = '```json\n{"key": "value"}\n```'
        assert truncation_reward(text) == 1.0

    def test_no_json_neutral(self):
        assert truncation_reward("Just plain text.") == 0.0

    def test_bare_json_complete(self):
        assert truncation_reward('{"name": "Alice"}') == 1.0

    def test_bare_json_unclosed_brace(self):
        assert truncation_reward('{"name": "Alice"') == -1.0

    def test_bare_json_unclosed_bracket(self):
        assert truncation_reward("[1, 2, 3") == -1.0

    def test_bare_json_trailing_comma(self):
        assert truncation_reward('{"a": 1,') == -1.0

    def test_bare_json_trailing_colon(self):
        assert truncation_reward('{"a":') == -1.0

    def test_bare_json_unterminated_string(self):
        assert truncation_reward('{"name": "Ali') == -1.0

    def test_bare_array_complete(self):
        assert truncation_reward("[1, 2, 3]") == 1.0

    def test_think_then_fenced_complete(self):
        text = (
            '<think>Some reasoning.</think>\n```json\n{"a": 1}\n```'
        )
        assert truncation_reward(text) == 1.0

    def test_think_then_bare_json_complete(self):
        text = '<think>Some reasoning.</think>\n{"a": 1}'
        assert truncation_reward(text) == 1.0

    def test_think_then_bare_json_truncated(self):
        text = '<think>Some reasoning.</think>\n{"a": 1,'
        assert truncation_reward(text) == -1.0

    def test_unclosed_fence_truncated(self):
        """Opening ```json fence without closing ``` → truncated."""
        text = 'Here is a JSON object:\n\n```json\n{"name": "John", "scores": [1, 2,'
        assert truncation_reward(text) == -1.0

    def test_prose_then_unclosed_fence(self):
        """Prose followed by unclosed fence with valid-looking JSON."""
        text = 'Here is the result:\n```json\n{"a": 1, "b": [1, 2, 3'
        assert truncation_reward(text) == -1.0


class TestRepetitionReward:
    def test_short_text_neutral(self):
        assert repetition_reward("short") == 1.0

    def test_normal_json(self):
        text = '```json\n{"name": "Alice", "age": 30, "city": "Rome", "job": "engineer"}\n```'
        assert repetition_reward(text) == 1.0

    def test_normal_array(self):
        items = ", ".join(
            f'{{"id": {i}, "val": "item_{i}"}}' for i in range(10)
        )
        text = f"```json\n[{items}]\n```"
        assert repetition_reward(text) == 1.0

    def test_severe_line_repetition(self):
        """Model stuck producing the same key-value pair."""
        lines = ['  "type": {"length": 5},'] * 60
        text = "```json\n{\n" + "\n".join(lines) + "\n}\n```"
        assert repetition_reward(text) == -1.0

    def test_severe_phrase_repetition(self):
        """Model stuck repeating a long phrase."""
        phrase = "ref: /path/to/documentation/"
        text = phrase * 50
        assert repetition_reward(text) == -1.0

    def test_moderate_repetition(self):
        """Some repetition but not extreme → 0.0."""
        # Build text with ~40% unique lines
        unique = [f'  "key_{i}": "value_{i}",' for i in range(8)]
        repeated = ['  "type": "default",'] * 12
        text = (
            "```json\n{\n" + "\n".join(unique + repeated) + "\n}\n```"
        )
        r = repetition_reward(text)
        assert r <= 0.0  # at least moderate penalty


class TestStrictnessReward:
    def test_code_block_only(self):
        text = '```json\n{"a": 1}\n```'
        assert strictness_reward(text) == 1.0

    def test_code_block_with_think(self):
        text = '<think>Let me think.</think>\n```json\n{"a": 1}\n```'
        assert strictness_reward(text) == 1.0

    def test_trivial_whitespace(self):
        text = '```json\n{"a": 1}\n```\n'
        assert strictness_reward(text) == 1.0

    def test_minor_preamble(self):
        """'Here:' is 5 chars → 0.5 (≤ 10 chars)."""
        text = 'Here:\n```json\n{"a": 1, "b": 2, "c": 3}\n```'
        assert strictness_reward(text) == 0.5

    def test_any_extra_text_penalized(self):
        """Even a short sentence after the block is penalized (> 10 chars)."""
        text = '```json\n{"a": 1}\n```\nThis is the answer.'
        assert strictness_reward(text) <= 0.0

    def test_significant_trailing_text(self):
        json_block = '```json\n{"a": 1}\n```'
        explanation = "\n\nThis JSON shows a simple object. " * 5
        text = json_block + explanation
        assert strictness_reward(text) <= -0.5

    def test_mostly_explanation(self):
        json_block = '```json\n{"x": 1}\n```'
        explanation = "\n" + "A required field. " * 30
        text = json_block + explanation
        assert strictness_reward(text) == -1.0

    def test_no_code_block(self):
        assert strictness_reward("just plain text") == 0.0


class TestBuildRewardFunctions:
    def test_returns_correct_count_with_thinking(self):
        funcs, weights = build_reward_functions(thinking=True)
        assert len(funcs) == 4
        assert len(weights) == 4
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_returns_three_without_thinking(self):
        funcs, weights = build_reward_functions(thinking=False)
        assert len(funcs) == 3
        assert len(weights) == 3
        names = [f.__name__ for f in funcs]
        assert "reasoning_reward" not in names

    def test_function_names_match_components(self):
        funcs, _ = build_reward_functions(thinking=True)
        names = [f.__name__ for f in funcs]
        assert "format_reward" in names
        assert "validity_reward" in names
        assert "schema_reward" in names
        assert "reasoning_reward" in names

    def test_each_function_returns_list(self):
        funcs, _ = build_reward_functions(thinking=True)
        completions = ['```json\n{"x": 1}\n```']
        for fn in funcs:
            result = fn(completions)
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], float)

    def test_thinking_false_redistributes_across_all_components(self):
        """When thinking=False, reasoning weight is split proportionally
        across ALL active components, including truncation."""
        funcs, weights = build_reward_functions(
            weight_format=0.25,
            weight_validity=0.30,
            weight_schema=0.30,
            weight_reasoning=0.10,
            weight_truncation=0.05,
            thinking=False,
        )
        names = [f.__name__ for f in funcs]
        assert "reasoning_reward" not in names
        assert len(funcs) == 4  # format, validity, schema, truncation
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_thinking_false_no_reasoning_weight_noop(self):
        """If reasoning weight is already 0, thinking=False changes nothing."""
        funcs, weights = build_reward_functions(
            weight_format=0.25,
            weight_validity=0.30,
            weight_schema=0.30,
            weight_reasoning=0.0,
            weight_truncation=0.15,
            thinking=False,
        )
        assert len(funcs) == 4
        assert weights == [0.25, 0.30, 0.30, 0.15]

    def test_repetition_included_when_weight_nonzero(self):
        funcs, weights = build_reward_functions(
            weight_format=0.15,
            weight_validity=0.20,
            weight_schema=0.25,
            weight_reasoning=0.30,
            weight_truncation=0.10,
            weight_repetition=0.10,
            thinking=True,
        )
        names = [f.__name__ for f in funcs]
        assert "repetition_reward" in names
        assert len(funcs) == 6

    def test_strictness_included_when_weight_nonzero(self):
        funcs, weights = build_reward_functions(
            weight_format=0.15,
            weight_validity=0.20,
            weight_schema=0.25,
            weight_reasoning=0.30,
            weight_truncation=0.10,
            weight_repetition=0.10,
            weight_strictness=0.10,
            thinking=True,
        )
        names = [f.__name__ for f in funcs]
        assert "strictness_reward" in names
        assert len(funcs) == 7
