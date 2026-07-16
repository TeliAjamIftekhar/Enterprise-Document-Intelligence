from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)

from src.book_config import load_book_config


DEFAULT_STANDARD_OUTPUT_CONFIG = Path(
    "workers/multimodal-ingestion/"
    "config/bda_document_project.json"
)

PROJECT_TYPE = "ASYNC"
MAXIMUM_STATUS_CHECKS = 60
POLL_INTERVAL_SECONDS = 5


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def derive_project_name(
    book_id: str,
    book_version: str,
) -> str:
    raw_name = (
        f"edi-{book_id}-{book_version}"
    ).lower()

    normalized = re.sub(
        r"[^a-z0-9-]+",
        "-",
        raw_name,
    )

    normalized = re.sub(
        r"-+",
        "-",
        normalized,
    ).strip("-")

    if len(normalized) <= 40:
        return normalized

    suffix = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:8]

    return (
        normalized[:31].rstrip("-")
        + "-"
        + suffix
    )


def load_standard_output_configuration(
    path: Path,
) -> dict[str, Any]:
    configuration = load_json_object(path)

    document = configuration.get(
        "document"
    )

    if not isinstance(document, dict):
        raise ValueError(
            "BDA standard output configuration "
            "must contain a document object."
        )

    return configuration


def build_client_token(
    *,
    project_name: str,
    project_stage: str,
    configuration: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "project_name": project_name,
            "project_stage": project_stage,
            "project_type": PROJECT_TYPE,
            "configuration": configuration,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def find_existing_projects(
    client: Any,
    *,
    project_name: str,
    project_stage: str,
) -> list[dict[str, Any]]:
    paginator = client.get_paginator(
        "list_data_automation_projects"
    )

    matches: list[dict[str, Any]] = []

    for page in paginator.paginate(
        projectStageFilter="ALL",
    ):
        for project in page.get(
            "projects",
            [],
        ):
            if (
                project.get("projectName")
                == project_name
                and project.get("projectStage")
                == project_stage
            ):
                matches.append(project)

    return matches


def get_project(
    client: Any,
    *,
    project_arn: str,
    project_stage: str,
) -> dict[str, Any]:
    return client.get_data_automation_project(
        projectArn=project_arn,
        projectStage=project_stage,
    )


def get_project_status(
    response: dict[str, Any],
) -> str:
    project = response.get("project")

    if not isinstance(project, dict):
        raise RuntimeError(
            "GetDataAutomationProject returned "
            "no project object."
        )

    status = project.get("status")

    if not isinstance(
        status,
        str,
    ) or not status:
        raise RuntimeError(
            "BDA project status is missing."
        )

    return status


def wait_for_project(
    client: Any,
    *,
    project_arn: str,
    project_stage: str,
) -> dict[str, Any]:
    for attempt in range(
        1,
        MAXIMUM_STATUS_CHECKS + 1,
    ):
        response = get_project(
            client,
            project_arn=project_arn,
            project_stage=project_stage,
        )

        status = get_project_status(
            response
        )

        print(
            f"Project status check "
            f"{attempt}: {status}"
        )

        if status == "COMPLETED":
            return response

        if status == "FAILED":
            raise RuntimeError(
                "BDA project creation failed: "
                + json.dumps(
                    response,
                    default=json_default,
                )
            )

        if status != "IN_PROGRESS":
            raise RuntimeError(
                "Unexpected BDA project status: "
                f"{status}"
            )

        time.sleep(
            POLL_INTERVAL_SECONDS
        )

    raise TimeoutError(
        "BDA project did not reach COMPLETED "
        "within the polling limit."
    )


def create_project(
    client: Any,
    *,
    project_name: str,
    project_description: str,
    project_stage: str,
    configuration: dict[str, Any],
) -> str:
    response = (
        client.create_data_automation_project(
            projectName=project_name,
            projectDescription=(
                project_description
            ),
            projectStage=project_stage,
            projectType=PROJECT_TYPE,
            standardOutputConfiguration=(
                configuration
            ),
            clientToken=build_client_token(
                project_name=project_name,
                project_stage=project_stage,
                configuration=configuration,
            ),
        )
    )

    project_arn = response.get(
        "projectArn"
    )

    if not isinstance(
        project_arn,
        str,
    ) or not project_arn:
        raise RuntimeError(
            "CreateDataAutomationProject "
            "returned no project ARN."
        )

    print(
        "Project creation request submitted."
    )
    print(
        "Initial status:",
        response.get("status"),
    )

    return project_arn


def find_configuration_mismatches(
    expected: Any,
    actual: Any,
    path: str = "standardOutputConfiguration",
) -> list[str]:
    mismatches: list[str] = []

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [
                f"{path}: expected object, "
                f"received "
                f"{type(actual).__name__}"
            ]

        for key, expected_value in (
            expected.items()
        ):
            child_path = f"{path}.{key}"

            if key not in actual:
                mismatches.append(
                    f"{child_path}: missing"
                )
                continue

            mismatches.extend(
                find_configuration_mismatches(
                    expected_value,
                    actual[key],
                    child_path,
                )
            )

        return mismatches

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [
                f"{path}: expected list, "
                f"received "
                f"{type(actual).__name__}"
            ]

        for value in expected:
            if value not in actual:
                mismatches.append(
                    f"{path}: missing value "
                    f"{value!r}"
                )

        return mismatches

    if expected != actual:
        mismatches.append(
            f"{path}: expected "
            f"{expected!r}, received "
            f"{actual!r}"
        )

    return mismatches


def validate_project(
    response: dict[str, Any],
    *,
    project_name: str,
    project_stage: str,
    expected_configuration: dict[str, Any],
) -> dict[str, Any]:
    project = response.get("project")

    if not isinstance(project, dict):
        raise RuntimeError(
            "BDA response contains no project."
        )

    expected_values = {
        "projectName": project_name,
        "projectStage": project_stage,
        "projectType": PROJECT_TYPE,
        "status": "COMPLETED",
    }

    for field, expected in (
        expected_values.items()
    ):
        actual = project.get(field)

        if actual != expected:
            raise RuntimeError(
                f"BDA project {field} mismatch: "
                f"expected={expected!r}, "
                f"actual={actual!r}"
            )

    mismatches = (
        find_configuration_mismatches(
            expected_configuration,
            project.get(
                "standardOutputConfiguration"
            ),
        )
    )

    if mismatches:
        raise RuntimeError(
            "BDA project configuration "
            "mismatch:\n- "
            + "\n- ".join(mismatches)
        )

    return project


def update_book_config(
    *,
    config_path: Path,
    project_arn: str,
) -> Path:
    raw_config = load_json_object(
        config_path
    )

    bda = raw_config.get("bda")

    if not isinstance(bda, dict):
        raise RuntimeError(
            "Book config contains no bda object."
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup_path = config_path.with_name(
        config_path.name
        + f".before-bda-{timestamp}.bak"
    )

    shutil.copy2(
        config_path,
        backup_path,
    )

    bda["project_arn"] = project_arn

    atomic_write_json(
        config_path,
        raw_config,
    )

    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or reuse a BDA project "
            "for a configured textbook."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--standard-output-config",
        type=Path,
        default=(
            DEFAULT_STANDARD_OUTPUT_CONFIG
        ),
    )

    parser.add_argument(
        "--project-name",
        default=None,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Create/reuse the AWS project and "
            "update the book configuration. "
            "Without this flag, only a local "
            "plan is produced."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_book_config(
        args.config
    )

    output_configuration = (
        load_standard_output_configuration(
            args.standard_output_config
        )
    )

    project_name = (
        args.project_name
        or derive_project_name(
            config.book.book_id,
            config.book.version,
        )
    )

    project_description = (
        "Multimodal textbook extraction for "
        f"{config.book.book_id} "
        f"{config.book.version}."
    )

    metadata_path = (
        Path(config.storage.local_root)
        / "bda-project.json"
    )

    print("=" * 68)
    print("CONFIG-DRIVEN BDA PROJECT")
    print("=" * 68)
    print(
        "Mode:          ",
        "EXECUTE"
        if args.execute
        else "LOCAL_PLAN",
    )
    print(
        "Config:        ",
        args.config,
    )
    print(
        "Region:        ",
        config.aws.region,
    )
    print(
        "Book ID:       ",
        config.book.book_id,
    )
    print(
        "Book version:  ",
        config.book.version,
    )
    print(
        "Project name:  ",
        project_name,
    )
    print(
        "Project stage: ",
        config.bda.stage,
    )
    print(
        "Project type:  ",
        PROJECT_TYPE,
    )
    print(
        "Profile ARN:   ",
        config.bda.profile_arn,
    )
    print(
        "Current ARN:   ",
        config.bda.project_arn,
    )
    print(
        "Metadata path: ",
        metadata_path,
    )
    print(
        "Output config: ",
        args.standard_output_config,
    )

    if not args.execute:
        print()
        print("=" * 68)
        print("LOCAL PLAN RESULT")
        print("=" * 68)
        print("Status:         LOCAL_VALIDATED")
        print("AWS writes:     0")
        print("Project created:False")
        print("Config updated: False")
        return 0

    client = boto3.client(
        "bedrock-data-automation",
        region_name=config.aws.region,
    )

    matches = find_existing_projects(
        client,
        project_name=project_name,
        project_stage=config.bda.stage,
    )

    if len(matches) > 1:
        raise RuntimeError(
            "More than one matching BDA "
            "project was found."
        )

    action = "reused"

    if matches:
        project_arn = str(
            matches[0]["projectArn"]
        )

        print()
        print(
            "Existing matching project found."
        )
        print(
            "Project ARN:",
            project_arn,
        )

        response = get_project(
            client,
            project_arn=project_arn,
            project_stage=config.bda.stage,
        )

        status = get_project_status(
            response
        )

        if status == "IN_PROGRESS":
            response = wait_for_project(
                client,
                project_arn=project_arn,
                project_stage=config.bda.stage,
            )

        elif status == "FAILED":
            raise RuntimeError(
                "Existing BDA project is FAILED."
            )

    else:
        action = "created"

        project_arn = create_project(
            client,
            project_name=project_name,
            project_description=(
                project_description
            ),
            project_stage=config.bda.stage,
            configuration=(
                output_configuration
            ),
        )

        response = wait_for_project(
            client,
            project_arn=project_arn,
            project_stage=config.bda.stage,
        )

    project = validate_project(
        response,
        project_name=project_name,
        project_stage=config.bda.stage,
        expected_configuration=(
            output_configuration
        ),
    )

    metadata = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "configuration": {
            "config_path": str(args.config),
            "standard_output_config": str(
                args.standard_output_config
            ),
        },
        "action": action,
        "project": project,
    }

    atomic_write_json(
        metadata_path,
        metadata,
    )

    backup_path = update_book_config(
        config_path=args.config,
        project_arn=project_arn,
    )

    print()
    print("=" * 68)
    print("BDA PROJECT READY")
    print("=" * 68)
    print("Status:       COMPLETED")
    print("Action:      ", action)
    print("Project ARN: ", project_arn)
    print("Project name:", project_name)
    print("Project type:", PROJECT_TYPE)
    print("Stage:       ", config.bda.stage)
    print("Metadata:    ", metadata_path)
    print("Config:      ", args.config)
    print("Backup:      ", backup_path)
    print("BDA invoked: False")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            "AWS error while configuring "
            f"BDA project: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"BDA project setup failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
