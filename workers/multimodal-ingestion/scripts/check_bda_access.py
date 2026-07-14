from __future__ import annotations

import json
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"
ACCOUNT_ID = "334590195171"

PROFILE_ARN = (
    f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:"
    "data-automation-profile/us.data-automation-v1"
)


def main() -> int:
    session = boto3.Session(region_name=REGION)
    available_services = set(session.get_available_services())

    required_services = {
        "bedrock-data-automation",
        "bedrock-data-automation-runtime",
    }

    missing = sorted(required_services - available_services)

    if missing:
        print(
            "Missing Boto3 service support:",
            ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    control_client = session.client(
        "bedrock-data-automation"
    )

    runtime_client = session.client(
        "bedrock-data-automation-runtime"
    )

    print("Region:", REGION)
    print("BDA profile ARN:", PROFILE_ARN)
    print(
        "Control client:",
        control_client.meta.service_model.service_name,
    )
    print(
        "Runtime client:",
        runtime_client.meta.service_model.service_name,
    )

    try:
        response = control_client.list_data_automation_projects(
            projectStageFilter="ALL",
            maxResults=10,
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})

        print(
            json.dumps(
                {
                    "status": "AWS_API_ERROR",
                    "code": error.get("Code"),
                    "message": error.get("Message"),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    except BotoCoreError as exc:
        print(
            f"Boto3 configuration error: {exc}",
            file=sys.stderr,
        )
        return 1

    projects = response.get("projects", [])

    print("BDA control-plane access: SUCCESS")
    print("Existing projects returned:", len(projects))

    for project in projects:
        print(
            "-",
            project.get("projectName"),
            project.get("projectArn"),
            project.get("projectStage"),
        )

    print("No document processing was invoked.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
