from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"

PROJECT_NAME = "textbook-multimodal-async-v1"
PROJECT_STAGE = "DEVELOPMENT"
PROJECT_TYPE = "ASYNC"

PROJECT_DESCRIPTION = (
    "Multimodal textbook extraction project with document, page, "
    "element, bounding-box, Markdown, CSV, and additional-file output."
)

CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/bda_document_project.json"
)

OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "bda-project.json"
)


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def load_standard_output_configuration() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"BDA configuration not found: {CONFIG_PATH}"
        )

    configuration = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    document = configuration.get("document")

    if not isinstance(document, dict):
        raise ValueError(
            "Configuration must contain a document object."
        )

    return configuration


def create_client():
    return boto3.client(
        "bedrock-data-automation",
        region_name=REGION,
    )


def find_existing_projects(
    client: Any,
) -> list[dict[str, Any]]:
    paginator = client.get_paginator(
        "list_data_automation_projects"
    )

    matches: list[dict[str, Any]] = []

    for page in paginator.paginate(
        projectStageFilter="ALL",
    ):
        for project in page.get("projects", []):
            if (
                project.get("projectName") == PROJECT_NAME
                and project.get("projectStage") == PROJECT_STAGE
            ):
                matches.append(project)

    return matches


def build_client_token(
    configuration: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "project_name": PROJECT_NAME,
            "project_stage": PROJECT_STAGE,
            "project_type": PROJECT_TYPE,
            "configuration": configuration,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def create_project(
    client: Any,
    configuration: dict[str, Any],
) -> str:
    response = client.create_data_automation_project(
        projectName=PROJECT_NAME,
        projectDescription=PROJECT_DESCRIPTION,
        projectStage=PROJECT_STAGE,
        projectType=PROJECT_TYPE,
        standardOutputConfiguration=configuration,
        clientToken=build_client_token(configuration),
    )

    project_arn = response.get("projectArn")

    if not project_arn:
        raise RuntimeError(
            "CreateDataAutomationProject returned no project ARN."
        )

    print("Project creation request submitted.")
    print("Initial status:", response.get("status"))

    return project_arn


def get_project(
    client: Any,
    project_arn: str,
) -> dict[str, Any]:
    return client.get_data_automation_project(
        projectArn=project_arn,
        projectStage=PROJECT_STAGE,
    )


def get_project_status(
    response: dict[str, Any],
) -> str:
    project = response.get("project")

    if not isinstance(project, dict):
        raise RuntimeError(
            "GetDataAutomationProject returned no project object."
        )

    status = project.get("status")

    if not isinstance(status, str) or not status:
        raise RuntimeError(
            "GetDataAutomationProject returned no project status. "
            f"Top-level keys: {list(response.keys())}; "
            f"project keys: {list(project.keys())}"
        )

    return status


def wait_for_project(
    client: Any,
    project_arn: str,
) -> dict[str, Any]:
    maximum_attempts = 60
    polling_interval_seconds = 5

    for attempt in range(1, maximum_attempts + 1):
        response = get_project(client, project_arn)
        status = get_project_status(response)

        print(
            f"Project status check {attempt}: {status}"
        )

        if status == "COMPLETED":
            return response

        if status == "FAILED":
            raise RuntimeError(
                "BDA project creation failed. "
                f"Response: {json.dumps(response, default=json_default)}"
            )

        if status != "IN_PROGRESS":
            raise RuntimeError(
                f"Unexpected BDA project status: {status}"
            )

        time.sleep(polling_interval_seconds)

    raise TimeoutError(
        "BDA project did not reach COMPLETED status "
        "within the configured polling attempts."
    )


def find_configuration_mismatches(
    expected: Any,
    actual: Any,
    path: str = "standardOutputConfiguration",
) -> list[str]:
    """
    Verify that every locally requested setting exists in the
    service-returned configuration.

    AWS may add default settings for modalities that were omitted
    from the create request, so extra service fields are accepted.
    """
    mismatches: list[str] = []

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [
                f"{path}: expected object, "
                f"received {type(actual).__name__}"
            ]

        for key, expected_value in expected.items():
            child_path = f"{path}.{key}"

            if key not in actual:
                mismatches.append(
                    f"{child_path}: missing from actual configuration"
                )
                continue

            mismatches.extend(
                find_configuration_mismatches(
                    expected=expected_value,
                    actual=actual[key],
                    path=child_path,
                )
            )

        return mismatches

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [
                f"{path}: expected list, "
                f"received {type(actual).__name__}"
            ]

        # Treat configuration lists as order-independent.
        missing_values = [
            item
            for item in expected
            if item not in actual
        ]

        if missing_values:
            mismatches.append(
                f"{path}: missing values {missing_values}; "
                f"actual values are {actual}"
            )

        return mismatches

    if expected != actual:
        mismatches.append(
            f"{path}: expected {expected!r}, "
            f"received {actual!r}"
        )

    return mismatches


def validate_project(
    response: dict[str, Any],
    expected_configuration: dict[str, Any],
) -> None:
    project = response.get("project")

    if not isinstance(project, dict):
        raise RuntimeError(
            "GetDataAutomationProject returned no project details."
        )

    if project.get("projectName") != PROJECT_NAME:
        raise RuntimeError("BDA project name does not match.")

    if project.get("projectStage") != PROJECT_STAGE:
        raise RuntimeError("BDA project stage does not match.")

    if project.get("projectType") != PROJECT_TYPE:
        raise RuntimeError("BDA project type does not match.")

    status = get_project_status(response)

    if status != "COMPLETED":
        raise RuntimeError(
            f"BDA project is not ready. Status: {status}"
        )

    actual_configuration = project.get(
        "standardOutputConfiguration"
    )

    mismatches = find_configuration_mismatches(
        expected=expected_configuration,
        actual=actual_configuration,
    )

    if mismatches:
        print("Expected configuration:")
        print(
            json.dumps(
                expected_configuration,
                indent=2,
                sort_keys=True,
            )
        )

        print("Actual configuration:")
        print(
            json.dumps(
                actual_configuration,
                indent=2,
                sort_keys=True,
            )
        )

        print("Configuration mismatches:")

        for mismatch in mismatches:
            print(f"- {mismatch}")

        raise RuntimeError(
            "The BDA project does not contain all locally "
            "requested configuration settings."
        )

    extra_modalities = sorted(
        set(actual_configuration or {})
        - set(expected_configuration)
    )

    print("Requested configuration validated successfully.")

    if extra_modalities:
        print(
            "Service-added default modalities accepted: "
            + ", ".join(extra_modalities)
        )


def save_project_metadata(
    response: dict[str, Any],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            response,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def main() -> int:
    configuration = load_standard_output_configuration()
    client = create_client()

    print("============================================")
    print("CREATING BDA PROJECT")
    print("============================================")
    print(f"Region:       {REGION}")
    print(f"Project name: {PROJECT_NAME}")
    print(f"Project type: {PROJECT_TYPE}")
    print(f"Stage:        {PROJECT_STAGE}")
    print()

    matches = find_existing_projects(client)

    if len(matches) > 1:
        raise RuntimeError(
            "More than one matching BDA project was found."
        )

    if matches:
        project_arn = matches[0]["projectArn"]

        print("Existing matching project found.")
        print(f"Project ARN: {project_arn}")

        response = get_project(
            client=client,
            project_arn=project_arn,
        )

        if get_project_status(response) == "IN_PROGRESS":
            response = wait_for_project(
                client=client,
                project_arn=project_arn,
            )

        elif get_project_status(response) == "FAILED":
            raise RuntimeError(
                "The existing BDA project is in FAILED status."
            )

    else:
        project_arn = create_project(
            client=client,
            configuration=configuration,
        )

        response = wait_for_project(
            client=client,
            project_arn=project_arn,
        )

    validate_project(
        response=response,
        expected_configuration=configuration,
    )

    save_project_metadata(response)

    project = response["project"]

    print()
    print("============================================")
    print("BDA PROJECT READY")
    print("============================================")
    print(f"Project ARN:  {project['projectArn']}")
    print(f"Project name: {project['projectName']}")
    print(f"Project type: {project['projectType']}")
    print(f"Project stage:{project['projectStage']}")
    print(f"Status:       {project['status']}")
    print(f"Metadata:     {OUTPUT_PATH}")
    print()
    print("No document processing was invoked.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS error while creating BDA project: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"BDA project creation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
