"""JSON prompt templates for synthetic dataset generation.

Templates are organized by difficulty level (simple, medium, hard).
Each template has an 'instruction' format string, a 'params' callable
that returns randomized parameters via the provided RNG, and an optional
'schema' callable that returns structural metadata for exact reward
validation (expected keys, toplevel type, counts, etc.).
"""

from __future__ import annotations

from typing import Any

# Type alias for a single template entry
Template = dict[str, Any]

# ---------------------------------------------------------------------------
# Simple — flat objects, single arrays, basic key-value pairs
# ---------------------------------------------------------------------------

SIMPLE: list[Template] = [
    {
        "instruction": (
            "Generate a valid JSON object with the following keys: "
            '"{k1}" (string), "{k2}" (integer), "{k3}" (boolean).'
        ),
        "params": lambda rng: {
            "k1": rng.choice(
                [
                    "name",
                    "title",
                    "label",
                    "city",
                    "color",
                    "brand",
                    "category",
                ]
            ),
            "k2": rng.choice(
                [
                    "age",
                    "count",
                    "score",
                    "year",
                    "quantity",
                    "rating",
                    "price",
                ]
            ),
            "k3": rng.choice(
                [
                    "active",
                    "verified",
                    "enabled",
                    "visible",
                    "published",
                    "premium",
                    "available",
                ]
            ),
        },
        "schema": lambda p: {
            "keys": [p["k1"], p["k2"], p["k3"]],
            "toplevel": "object",
            "key_types": {
                p["k1"]: "string",
                p["k2"]: "integer",
                p["k3"]: "boolean",
            },
        },
    },
    {
        "instruction": "Generate a JSON array containing exactly {n} strings representing {topic}.",
        "params": lambda rng: {
            "n": rng.randint(3, 7),
            "topic": rng.choice(
                [
                    "fruit names",
                    "country names",
                    "programming languages",
                    "animal species",
                    "planet names",
                    "car brands",
                    "musical instruments",
                    "European capital cities",
                    "file extensions",
                    "color names",
                ]
            ),
        },
        "schema": lambda p: {
            "toplevel": "array",
            "count": p["n"],
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a key "{k1}" (string) and a key '
            '"{k2}" which is an array of {n} integers.'
        ),
        "params": lambda rng: {
            "k1": rng.choice(["id", "name", "label", "code", "tag"]),
            "k2": rng.choice(
                ["values", "scores", "data", "items", "measurements"]
            ),
            "n": rng.randint(3, 6),
        },
        "schema": lambda p: {
            "keys": [p["k1"], p["k2"]],
            "toplevel": "object",
            "count": p["n"],
            "array_key": p["k2"],
            "key_types": {p["k1"]: "string", p["k2"]: "array"},
        },
    },
    {
        "instruction": (
            "Generate a JSON object with exactly {n} key-value pairs where all "
            "values are of type {vtype}. Use descriptive key names related to {topic}."
        ),
        "params": lambda rng: {
            "n": rng.randint(3, 6),
            "vtype": rng.choice(["string", "integer", "boolean"]),
            "topic": rng.choice(
                [
                    "weather",
                    "food",
                    "sports",
                    "music",
                    "travel",
                    "health",
                ]
            ),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "count": p["n"],
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a simple {entity} "
            'with keys "{k1}" (string), "{k2}" ({t2}), and "{k3}" ({t3}).'
        ),
        "params": lambda rng: {
            "entity": rng.choice(
                [
                    "contact card",
                    "book entry",
                    "to-do item",
                    "event",
                    "bookmark",
                    "notification",
                    "tag",
                    "setting",
                    "preference",
                ]
            ),
            "k1": rng.choice(
                ["name", "title", "description", "label"]
            ),
            "k2": rng.choice(["priority", "order", "level", "index"]),
            "t2": rng.choice(["integer", "number"]),
            "k3": rng.choice(
                ["done", "read", "starred", "archived", "pinned"]
            ),
            "t3": "boolean",
        },
        "schema": lambda p: {
            "keys": [p["k1"], p["k2"], p["k3"]],
            "toplevel": "object",
            "key_types": {
                p["k1"]: "string",
                p["k2"]: p["t2"],
                p["k3"]: p["t3"],
            },
        },
    },
    {
        "instruction": (
            "Generate a JSON array of exactly {n} objects, each containing "
            'only a "{k1}" (string) and a "{k2}" (integer).'
        ),
        "params": lambda rng: {
            "n": rng.randint(2, 5),
            "k1": rng.choice(
                ["name", "item", "label", "title", "city"]
            ),
            "k2": rng.choice(
                ["value", "count", "score", "amount", "population"]
            ),
        },
        "schema": lambda p: {
            "toplevel": "array",
            "count": p["n"],
            "item_keys": [p["k1"], p["k2"]],
            "item_key_types": {p["k1"]: "string", p["k2"]: "integer"},
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a "{k1}" key (string) and a "{k2}" key '
            "containing a flat array of {n} {elem_type}."
        ),
        "params": lambda rng: {
            "k1": rng.choice(
                ["category", "group", "type", "section"]
            ),
            "k2": rng.choice(
                ["items", "elements", "entries", "members"]
            ),
            "n": rng.randint(3, 7),
            "elem_type": rng.choice(
                ["strings", "integers", "booleans"]
            ),
        },
        "schema": lambda p: {
            "keys": [p["k1"], p["k2"]],
            "toplevel": "object",
            "count": p["n"],
            "array_key": p["k2"],
            "key_types": {p["k1"]: "string", p["k2"]: "array"},
        },
    },
    {
        "instruction": (
            "Generate a valid JSON object representing a key-value mapping "
            "of {n} {domain} abbreviations to their full names."
        ),
        "params": lambda rng: {
            "n": rng.randint(3, 6),
            "domain": rng.choice(
                [
                    "country",
                    "US state",
                    "HTTP status code",
                    "chemical element",
                    "currency",
                    "time zone",
                    "unit of measurement",
                ]
            ),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "count": p["n"],
        },
    },
]

# ---------------------------------------------------------------------------
# Medium — nested objects, mixed types, multiple arrays
# ---------------------------------------------------------------------------

MEDIUM: list[Template] = [
    {
        "instruction": (
            "Generate a JSON array of {n} objects, each with keys "
            '"{k1}" (string), "{k2}" (integer), and a nested object '
            '"{k3}" containing "{nk1}" (string) and "{nk2}" (string).'
        ),
        "params": lambda rng: {
            "n": rng.randint(2, 4),
            "k1": rng.choice(
                ["name", "title", "username", "product", "label"]
            ),
            "k2": rng.choice(
                ["age", "id", "score", "quantity", "price"]
            ),
            "k3": rng.choice(
                [
                    "address",
                    "contact",
                    "location",
                    "details",
                    "metadata",
                ]
            ),
            "nk1": rng.choice(
                ["street", "city", "email", "line1", "type"]
            ),
            "nk2": rng.choice(
                ["zip", "country", "phone", "region", "code"]
            ),
        },
        "schema": lambda p: {
            "toplevel": "array",
            "count": p["n"],
            "item_keys": [p["k1"], p["k2"], p["k3"]],
            "item_nested_keys": {p["k3"]: [p["nk1"], p["nk2"]]},
            "item_key_types": {
                p["k1"]: "string",
                p["k2"]: "integer",
                p["k3"]: "object",
            },
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with at least {n} fields, "
            "including one nested object and one array field."
        ),
        "params": lambda rng: {
            "entity": rng.choice(
                [
                    "user profile",
                    "product listing",
                    "blog post",
                    "movie record",
                    "employee record",
                    "restaurant menu item",
                    "flight booking",
                    "music album",
                    "course syllabus",
                    "recipe",
                ]
            ),
            "n": rng.randint(5, 8),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "min_count": p["n"],
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a configuration file for a {app_type} "
            "with keys for {k1}, {k2}, and a nested {k3} section."
        ),
        "params": lambda rng: {
            "app_type": rng.choice(
                [
                    "web server",
                    "database",
                    "logging service",
                    "cache layer",
                    "message queue",
                    "API gateway",
                    "monitoring agent",
                ]
            ),
            "k1": rng.choice(["host", "port", "name", "endpoint"]),
            "k2": rng.choice(
                [
                    "timeout",
                    "retries",
                    "max_connections",
                    "buffer_size",
                ]
            ),
            "k3": rng.choice(
                ["auth", "ssl", "logging", "metrics", "cors"]
            ),
        },
        "schema": lambda p: {
            "keys": [p["k1"], p["k2"], p["k3"]],
            "toplevel": "object",
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} that includes "
            "a list of {n} {sub_entity}, each with at least 3 fields."
        ),
        "params": lambda rng: {
            "entity": rng.choice(
                [
                    "classroom",
                    "shopping cart",
                    "playlist",
                    "project board",
                    "team roster",
                    "menu",
                    "itinerary",
                    "inventory",
                ]
            ),
            "n": rng.randint(2, 5),
            "sub_entity": rng.choice(
                [
                    "students",
                    "items",
                    "tracks",
                    "tasks",
                    "members",
                    "dishes",
                    "stops",
                    "products",
                ]
            ),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "count": p["n"],
            "array_key": p["sub_entity"],
            "item_min_count": 3,
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {doc_type} with the following "
            'sections: "{s1}", "{s2}", and "{s3}". Each section should be a nested '
            "object with at least 2 fields."
        ),
        "params": lambda rng: {
            "doc_type": rng.choice(
                [
                    "report",
                    "invoice",
                    "contract",
                    "specification",
                    "manifest",
                    "log entry",
                    "audit record",
                ]
            ),
            "s1": rng.choice(
                ["header", "metadata", "summary", "info"]
            ),
            "s2": rng.choice(
                ["body", "content", "details", "payload"]
            ),
            "s3": rng.choice(
                ["footer", "signature", "status", "notes"]
            ),
        },
        "schema": lambda p: {
            "keys": [p["s1"], p["s2"], p["s3"]],
            "toplevel": "object",
            "nested_min_count": {p["s1"]: 2, p["s2"]: 2, p["s3"]: 2},
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a "{k1}" key (string), a "{k2}" key '
            '(ISO 8601 date string), and a "{k3}" key containing an array of '
            'objects with "{nk1}" (string) and "{nk2}" (number) fields.'
        ),
        "params": lambda rng: {
            "k1": rng.choice(["title", "name", "event", "project"]),
            "k2": rng.choice(
                ["created_at", "due_date", "start_date", "timestamp"]
            ),
            "k3": rng.choice(
                ["entries", "line_items", "records", "milestones"]
            ),
            "nk1": rng.choice(
                ["description", "label", "name", "note"]
            ),
            "nk2": rng.choice(
                ["amount", "hours", "progress", "value"]
            ),
        },
        "schema": lambda p: {
            "keys": [p["k1"], p["k2"], p["k3"]],
            "toplevel": "object",
            "array_key": p["k3"],
            "item_keys": [p["nk1"], p["nk2"]],
            "key_types": {
                p["k1"]: "string",
                p["k2"]: "string",
                p["k3"]: "array",
            },
            "item_key_types": {
                p["nk1"]: "string",
                p["nk2"]: "number",
            },
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {form_type} form schema with {n} fields. "
            'Each field should have "label" (string), "type" (one of "text", "number", '
            '"email", "select"), and "required" (boolean).'
        ),
        "params": lambda rng: {
            "form_type": rng.choice(
                [
                    "registration",
                    "contact",
                    "survey",
                    "feedback",
                    "checkout",
                    "onboarding",
                    "support ticket",
                ]
            ),
            "n": rng.randint(4, 7),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "count": p["n"],
            "item_keys": ["label", "type", "required"],
            "item_enums": {
                "type": ["text", "number", "email", "select"]
            },
            "item_key_types": {
                "label": "string",
                "type": "string",
                "required": "boolean",
            },
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with an "
            '"id" (integer), "name" (string), "tags" (array of strings), '
            'and a "metadata" nested object with at least 3 key-value pairs.'
        ),
        "params": lambda rng: {
            "entity": rng.choice(
                [
                    "document",
                    "asset",
                    "resource",
                    "artifact",
                    "dataset",
                    "model",
                    "deployment",
                    "experiment",
                ]
            ),
        },
        "schema": lambda p: {
            "keys": ["id", "name", "tags", "metadata"],
            "toplevel": "object",
            "nested_min_count": {"metadata": 3},
            "key_types": {
                "id": "integer",
                "name": "string",
                "tags": "array",
                "metadata": "object",
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Hard — deep nesting, schemas, API specs, workflows
# ---------------------------------------------------------------------------

HARD: list[Template] = [
    {
        "instruction": (
            "Generate a JSON Schema (draft-07) that validates objects representing "
            "a {entity}. The schema must include required fields, type constraints, "
            "a nested object property, and an array property with item validation."
        ),
        "params": lambda rng: {
            "entity": rng.choice(
                [
                    "REST API error response",
                    "e-commerce order",
                    "weather forecast",
                    "user registration form",
                    "CI/CD pipeline definition",
                    "IoT sensor reading",
                    "GraphQL query result",
                    "OAuth2 token response",
                ]
            ),
        },
        "schema": lambda p: {
            "toplevel": "object",
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a paginated API response for a "
            "list of {entity}. Include metadata fields (page, per_page, total, "
            "total_pages) and a results array with {n} items, each having at "
            "least {f} fields including one nested object."
        ),
        "params": lambda rng: {
            "entity": rng.choice(
                [
                    "users",
                    "products",
                    "articles",
                    "transactions",
                    "repositories",
                    "notifications",
                    "search results",
                ]
            ),
            "n": rng.randint(2, 4),
            "f": rng.randint(4, 6),
        },
        "schema": lambda p: {
            "keys": [
                "page",
                "per_page",
                "total",
                "total_pages",
                "results",
            ],
            "toplevel": "object",
            "count": p["n"],
            "array_key": "results",
            "item_min_count": p["f"],
            "key_types": {
                "page": "integer",
                "per_page": "integer",
                "total": "integer",
                "total_pages": "integer",
                "results": "array",
            },
        },
    },
    {
        "instruction": (
            "Generate a deeply nested JSON object (at least 4 levels of nesting) "
            "representing a {domain} hierarchy. Each level should have a "
            '"name" (string), "id" (integer), and "children" (array of sub-objects).'
        ),
        "params": lambda rng: {
            "domain": rng.choice(
                [
                    "company organizational chart",
                    "file system directory tree",
                    "category taxonomy",
                    "geographical region breakdown",
                    "menu navigation structure",
                    "permission role hierarchy",
                ]
            ),
        },
        "schema": lambda p: {
            "keys": ["name", "id", "children"],
            "toplevel": "object",
            "depth": 4,
            "key_types": {
                "name": "string",
                "id": "integer",
                "children": "array",
            },
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a full {api_type} API endpoint "
            "specification (OpenAPI-style) including path, method, parameters "
            "(with types and required flags), request body schema, and "
            "response schema with example values."
        ),
        "params": lambda rng: {
            "api_type": rng.choice(
                [
                    "user management",
                    "payment processing",
                    "search",
                    "file upload",
                    "authentication",
                    "notification",
                ]
            ),
        },
        "schema": lambda p: {
            "toplevel": "object",
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {workflow} workflow definition "
            'with at least {n} steps. Each step must have an "id", "name", '
            '"type" (one of "action", "condition", "loop"), "config" '
            '(nested object), and "next" (string or null).'
        ),
        "params": lambda rng: {
            "workflow": rng.choice(
                [
                    "data pipeline",
                    "CI/CD build",
                    "order fulfillment",
                    "user onboarding",
                    "content moderation",
                    "ETL process",
                ]
            ),
            "n": rng.randint(4, 6),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "min_count": p["n"],
            "item_keys": ["id", "name", "type", "config", "next"],
            "item_enums": {"type": ["action", "condition", "loop"]},
            "item_key_types": {"config": "object"},
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a dashboard configuration with "
            '{n} widgets. Each widget has "id", "type" (chart, table, metric, '
            'or map), "title", "position" (object with x, y, w, h), and a '
            '"data_source" object with "endpoint" (string), "params" (object), '
            'and "refresh_interval" (integer).'
        ),
        "params": lambda rng: {
            "n": rng.randint(3, 6),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "count": p["n"],
            "item_keys": [
                "id",
                "type",
                "title",
                "position",
                "data_source",
            ],
            "item_enums": {
                "type": ["chart", "table", "metric", "map"]
            },
            "item_nested_keys": {
                "position": ["x", "y", "w", "h"],
                "data_source": [
                    "endpoint",
                    "params",
                    "refresh_interval",
                ],
            },
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {domain} event conforming "
            "to the CloudEvents specification (version 1.0). Include specversion, "
            "id, source, type, subject, time, datacontenttype, and a nested data "
            "object with at least {n} domain-specific fields."
        ),
        "params": lambda rng: {
            "domain": rng.choice(
                [
                    "e-commerce purchase",
                    "IoT temperature alert",
                    "user authentication",
                    "deployment completed",
                    "payment failed",
                    "inventory low",
                ]
            ),
            "n": rng.randint(4, 7),
        },
        "schema": lambda p: {
            "keys": [
                "specversion",
                "id",
                "source",
                "type",
                "subject",
                "time",
                "datacontenttype",
                "data",
            ],
            "toplevel": "object",
            "nested_min_count": {"data": p["n"]},
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a database migration plan with "
            '{n} operations. Each operation should be an object with "order" '
            '(integer), "type" (one of "create_table", "add_column", '
            '"create_index", "add_constraint"), "table" (string), and '
            '"definition" (nested object describing the change in detail).'
        ),
        "params": lambda rng: {
            "n": rng.randint(3, 6),
        },
        "schema": lambda p: {
            "toplevel": "object",
            "count": p["n"],
            "item_keys": ["order", "type", "table", "definition"],
            "item_enums": {
                "type": [
                    "create_table",
                    "add_column",
                    "create_index",
                    "add_constraint",
                ]
            },
            "item_key_types": {
                "order": "integer",
                "type": "string",
                "table": "string",
                "definition": "object",
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Registry and weights
# ---------------------------------------------------------------------------

POOLS: dict[str, dict[str, str | list[Template]]] = {
    "json_simple": {"templates": SIMPLE, "difficulty": "simple"},
    "json_medium": {"templates": MEDIUM, "difficulty": "medium"},
    "json_hard": {"templates": HARD, "difficulty": "hard"},
}

DIFFICULTY_WEIGHTS: dict[str, float] = {
    "simple": 0.10,
    "medium": 0.30,
    "hard": 0.60,
}
