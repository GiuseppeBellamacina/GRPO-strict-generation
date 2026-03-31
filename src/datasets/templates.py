"""JSON prompt templates for synthetic dataset generation.

Templates are organized by difficulty level (simple, medium, hard).
Each template has an 'instruction' format string and a 'params' callable
that returns randomized parameters via the provided RNG.
"""

from __future__ import annotations

import random
from typing import Any, Callable

# Type alias for a single template entry
Template = dict[str, str | Callable[[], dict[str, Any]]]

# ---------------------------------------------------------------------------
# Simple — flat objects, single arrays, basic key-value pairs
# ---------------------------------------------------------------------------

SIMPLE: list[Template] = [
    {
        "instruction": (
            "Generate a valid JSON object with the following keys: "
            '"{k1}" (string), "{k2}" (integer), "{k3}" (boolean).'
        ),
        "params": lambda: {
            "k1": random.choice(
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
            "k2": random.choice(
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
            "k3": random.choice(
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
    },
    {
        "instruction": "Generate a JSON array containing exactly {n} strings representing {topic}.",
        "params": lambda: {
            "n": random.randint(3, 7),
            "topic": random.choice(
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
    },
    {
        "instruction": (
            'Generate a JSON object with a key "{k1}" (string) and a key '
            '"{k2}" which is an array of {n} integers.'
        ),
        "params": lambda: {
            "k1": random.choice(
                ["id", "name", "label", "code", "tag"]
            ),
            "k2": random.choice(
                ["values", "scores", "data", "items", "measurements"]
            ),
            "n": random.randint(3, 6),
        },
    },
    {
        "instruction": (
            "Generate a JSON object with exactly {n} key-value pairs where all "
            "values are of type {vtype}. Use descriptive key names related to {topic}."
        ),
        "params": lambda: {
            "n": random.randint(3, 6),
            "vtype": random.choice(["string", "integer", "boolean"]),
            "topic": random.choice(
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
    },
    {
        "instruction": (
            "Generate a JSON object representing a simple {entity} "
            'with keys "{k1}" (string), "{k2}" ({t2}), and "{k3}" ({t3}).'
        ),
        "params": lambda: {
            "entity": random.choice(
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
            "k1": random.choice(
                ["name", "title", "description", "label"]
            ),
            "k2": random.choice(
                ["priority", "order", "level", "index"]
            ),
            "t2": random.choice(["integer", "number"]),
            "k3": random.choice(
                ["done", "read", "starred", "archived", "pinned"]
            ),
            "t3": "boolean",
        },
    },
    {
        "instruction": (
            "Generate a JSON array of exactly {n} objects, each containing "
            'only a "{k1}" (string) and a "{k2}" (integer).'
        ),
        "params": lambda: {
            "n": random.randint(2, 5),
            "k1": random.choice(
                ["name", "item", "label", "title", "city"]
            ),
            "k2": random.choice(
                ["value", "count", "score", "amount", "population"]
            ),
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a "{k1}" key (string) and a "{k2}" key '
            "containing a flat array of {n} {elem_type}."
        ),
        "params": lambda: {
            "k1": random.choice(
                ["category", "group", "type", "section"]
            ),
            "k2": random.choice(
                ["items", "elements", "entries", "members"]
            ),
            "n": random.randint(3, 7),
            "elem_type": random.choice(
                ["strings", "integers", "booleans"]
            ),
        },
    },
    {
        "instruction": (
            "Generate a valid JSON object representing a key-value mapping "
            "of {n} {domain} abbreviations to their full names."
        ),
        "params": lambda: {
            "n": random.randint(3, 6),
            "domain": random.choice(
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
        "params": lambda: {
            "n": random.randint(2, 4),
            "k1": random.choice(
                ["name", "title", "username", "product", "label"]
            ),
            "k2": random.choice(
                ["age", "id", "score", "quantity", "price"]
            ),
            "k3": random.choice(
                [
                    "address",
                    "contact",
                    "location",
                    "details",
                    "metadata",
                ]
            ),
            "nk1": random.choice(
                ["street", "city", "email", "line1", "type"]
            ),
            "nk2": random.choice(
                ["zip", "country", "phone", "region", "code"]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with at least {n} fields, "
            "including one nested object and one array field."
        ),
        "params": lambda: {
            "entity": random.choice(
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
            "n": random.randint(5, 8),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a configuration file for a {app_type} "
            "with keys for {k1}, {k2}, and a nested {k3} section."
        ),
        "params": lambda: {
            "app_type": random.choice(
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
            "k1": random.choice(["host", "port", "name", "endpoint"]),
            "k2": random.choice(
                [
                    "timeout",
                    "retries",
                    "max_connections",
                    "buffer_size",
                ]
            ),
            "k3": random.choice(
                ["auth", "ssl", "logging", "metrics", "cors"]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} that includes "
            "a list of {n} {sub_entity}, each with at least 3 fields."
        ),
        "params": lambda: {
            "entity": random.choice(
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
            "n": random.randint(2, 5),
            "sub_entity": random.choice(
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
    },
    {
        "instruction": (
            "Generate a JSON object representing a {doc_type} with the following "
            'sections: "{s1}", "{s2}", and "{s3}". Each section should be a nested '
            "object with at least 2 fields."
        ),
        "params": lambda: {
            "doc_type": random.choice(
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
            "s1": random.choice(
                ["header", "metadata", "summary", "info"]
            ),
            "s2": random.choice(
                ["body", "content", "details", "payload"]
            ),
            "s3": random.choice(
                ["footer", "signature", "status", "notes"]
            ),
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a "{k1}" key (string), a "{k2}" key '
            '(ISO 8601 date string), and a "{k3}" key containing an array of '
            'objects with "{nk1}" (string) and "{nk2}" (number) fields.'
        ),
        "params": lambda: {
            "k1": random.choice(
                ["title", "name", "event", "project"]
            ),
            "k2": random.choice(
                ["created_at", "due_date", "start_date", "timestamp"]
            ),
            "k3": random.choice(
                ["entries", "line_items", "records", "milestones"]
            ),
            "nk1": random.choice(
                ["description", "label", "name", "note"]
            ),
            "nk2": random.choice(
                ["amount", "hours", "progress", "value"]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {form_type} form schema with {n} fields. "
            'Each field should have "label" (string), "type" (one of "text", "number", '
            '"email", "select"), and "required" (boolean).'
        ),
        "params": lambda: {
            "form_type": random.choice(
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
            "n": random.randint(4, 7),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with an "
            '"id" (integer), "name" (string), "tags" (array of strings), '
            'and a "metadata" nested object with at least 3 key-value pairs.'
        ),
        "params": lambda: {
            "entity": random.choice(
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
        "params": lambda: {
            "entity": random.choice(
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
    },
    {
        "instruction": (
            "Generate a JSON object representing a paginated API response for a "
            "list of {entity}. Include metadata fields (page, per_page, total, "
            "total_pages) and a results array with {n} items, each having at "
            "least {f} fields including one nested object."
        ),
        "params": lambda: {
            "entity": random.choice(
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
            "n": random.randint(2, 4),
            "f": random.randint(4, 6),
        },
    },
    {
        "instruction": (
            "Generate a deeply nested JSON object (at least 4 levels of nesting) "
            "representing a {domain} hierarchy. Each level should have a "
            '"name" (string), "id" (integer), and "children" (array of sub-objects).'
        ),
        "params": lambda: {
            "domain": random.choice(
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
    },
    {
        "instruction": (
            "Generate a JSON object representing a full {api_type} API endpoint "
            "specification (OpenAPI-style) including path, method, parameters "
            "(with types and required flags), request body schema, and "
            "response schema with example values."
        ),
        "params": lambda: {
            "api_type": random.choice(
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
    },
    {
        "instruction": (
            "Generate a JSON object representing a {workflow} workflow definition "
            'with at least {n} steps. Each step must have an "id", "name", '
            '"type" (one of "action", "condition", "loop"), "config" '
            '(nested object), and "next" (string or null).'
        ),
        "params": lambda: {
            "workflow": random.choice(
                [
                    "data pipeline",
                    "CI/CD build",
                    "order fulfillment",
                    "user onboarding",
                    "content moderation",
                    "ETL process",
                ]
            ),
            "n": random.randint(4, 6),
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
        "params": lambda: {
            "n": random.randint(3, 6),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {domain} event conforming "
            "to the CloudEvents specification (version 1.0). Include specversion, "
            "id, source, type, subject, time, datacontenttype, and a nested data "
            "object with at least {n} domain-specific fields."
        ),
        "params": lambda: {
            "domain": random.choice(
                [
                    "e-commerce purchase",
                    "IoT temperature alert",
                    "user authentication",
                    "deployment completed",
                    "payment failed",
                    "inventory low",
                ]
            ),
            "n": random.randint(4, 7),
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
        "params": lambda: {
            "n": random.randint(3, 6),
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
