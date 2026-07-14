from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)


REGION = "us-east-1"
EXPECTED_ACCOUNT_ID = "334590195171"

COLLECTION_NAME = "edi-textbook-vector"
INDEX_NAME = "grade-9-english-kaveri-v1"

OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "preflight-report.json"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_response(
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.items()
        if key != "ResponseMetadata"
    }


def paginate(
    client: Any,
    operation_name: str,
    result_key: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    operation = getattr(
        client,
        operation_name,
    )

    request = dict(kwargs)

    results: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()

    while True:
        response = operation(**request)

        values = response.get(
            result_key,
            [],
        )

        if isinstance(values, list):
            results.extend(values)

        next_token = response.get(
            "nextToken"
        )

        if not isinstance(
            next_token,
            str,
        ) or not next_token:
            break

        if next_token in seen_tokens:
            raise RuntimeError(
                "Repeated pagination token returned "
                f"by {operation_name}."
            )

        seen_tokens.add(next_token)

        request["nextToken"] = (
            next_token
        )

    return results


def list_collections(
    client: Any,
) -> list[dict[str, Any]]:
    return paginate(
        client=client,
        operation_name="list_collections",
        result_key="collectionSummaries",
    )


def list_security_policies(
    client: Any,
    policy_type: str,
) -> list[dict[str, Any]]:
    return paginate(
        client=client,
        operation_name="list_security_policies",
        result_key="securityPolicySummaries",
        type=policy_type,
    )


def list_access_policies(
    client: Any,
) -> list[dict[str, Any]]:
    return paginate(
        client=client,
        operation_name="list_access_policies",
        result_key="accessPolicySummaries",
        type="data",
    )


def list_lifecycle_policies(
    client: Any,
) -> list[dict[str, Any]]:
    try:
        return paginate(
            client=client,
            operation_name="list_lifecycle_policies",
            result_key="lifecyclePolicySummaries",
            type="retention",
        )

    except (
        ClientError,
        BotoCoreError,
        KeyError,
    ) as exc:
        print(
            "Lifecycle-policy listing warning:",
            exc,
        )

        return []


def get_collection_details(
    client: Any,
    collections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collection_ids = [
        str(collection["id"])
        for collection in collections
        if isinstance(collection.get("id"), str)
        and collection["id"]
    ]

    if not collection_ids:
        return []

    details: list[dict[str, Any]] = []

    for start in range(
        0,
        len(collection_ids),
        20,
    ):
        batch = collection_ids[
            start:start + 20
        ]

        response = client.batch_get_collection(
            ids=batch
        )

        batch_details = response.get(
            "collectionDetails",
            [],
        )

        if isinstance(batch_details, list):
            details.extend(batch_details)

    return details


def find_name_matches(
    records: list[dict[str, Any]],
    expected_name: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("name") == expected_name
    ]


def main() -> int:
    print("============================================")
    print("OPENSEARCH SERVERLESS PREFLIGHT")
    print("============================================")
    print(f"Region:              {REGION}")
    print(f"Target collection:   {COLLECTION_NAME}")
    print(f"Target index:        {INDEX_NAME}")
    print()

    sts_client = boto3.client(
        "sts",
        region_name=REGION,
    )

    aoss_client = boto3.client(
        "opensearchserverless",
        region_name=REGION,
    )

    identity = clean_response(
        sts_client.get_caller_identity()
    )

    account_id = str(
        identity.get("Account", "")
    )

    caller_arn = str(
        identity.get("Arn", "")
    )

    print(f"AWS account:         {account_id}")
    print(f"Caller ARN:          {caller_arn}")

    if account_id != EXPECTED_ACCOUNT_ID:
        raise RuntimeError(
            "AWS account does not match the project "
            f"account. Expected={EXPECTED_ACCOUNT_ID}, "
            f"actual={account_id}"
        )

    collections = list_collections(
        aoss_client
    )

    collection_details = (
        get_collection_details(
            client=aoss_client,
            collections=collections,
        )
    )

    encryption_policies = (
        list_security_policies(
            client=aoss_client,
            policy_type="encryption",
        )
    )

    network_policies = (
        list_security_policies(
            client=aoss_client,
            policy_type="network",
        )
    )

    access_policies = (
        list_access_policies(
            aoss_client
        )
    )

    lifecycle_policies = (
        list_lifecycle_policies(
            aoss_client
        )
    )

    target_collections = find_name_matches(
        collections,
        COLLECTION_NAME,
    )

    print()
    print("Current account inventory:")
    print(
        f"Collections:          "
        f"{len(collections)}"
    )
    print(
        f"Encryption policies: "
        f"{len(encryption_policies)}"
    )
    print(
        f"Network policies:    "
        f"{len(network_policies)}"
    )
    print(
        f"Data access policies:"
        f" {len(access_policies)}"
    )
    print(
        f"Lifecycle policies:  "
        f"{len(lifecycle_policies)}"
    )

    print()
    print("Collections:")

    if collections:
        for collection in collections:
            print(
                "- "
                f"name={collection.get('name')} | "
                f"id={collection.get('id')} | "
                f"type={collection.get('type')} | "
                f"status={collection.get('status')}"
            )
    else:
        print("- None")

    print()
    print("Existing policy names:")

    for label, policies in (
        (
            "encryption",
            encryption_policies,
        ),
        (
            "network",
            network_policies,
        ),
        (
            "data",
            access_policies,
        ),
        (
            "retention",
            lifecycle_policies,
        ),
    ):
        names = sorted(
            str(policy.get("name"))
            for policy in policies
            if policy.get("name")
        )

        print(
            f"- {label}: "
            + (
                ", ".join(names)
                if names
                else "None"
            )
        )

    conflicts: list[str] = []

    if len(target_collections) > 1:
        conflicts.append(
            "Multiple collections exist with the "
            f"target name {COLLECTION_NAME}."
        )

    target_exists = (
        len(target_collections) == 1
    )

    if target_exists:
        target = target_collections[0]

        if target.get("type") != "VECTORSEARCH":
            conflicts.append(
                "The target collection name already "
                "exists but is not VECTORSEARCH."
            )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "region": REGION,
        "expected_account_id": (
            EXPECTED_ACCOUNT_ID
        ),
        "identity": identity,
        "target": {
            "collection_name": (
                COLLECTION_NAME
            ),
            "collection_type": (
                "VECTORSEARCH"
            ),
            "index_name": INDEX_NAME,
            "vector_dimensions": 1024,
            "target_collection_exists": (
                target_exists
            ),
        },
        "inventory": {
            "collections": collections,
            "collection_details": (
                collection_details
            ),
            "encryption_policies": (
                encryption_policies
            ),
            "network_policies": (
                network_policies
            ),
            "data_access_policies": (
                access_policies
            ),
            "lifecycle_policies": (
                lifecycle_policies
            ),
        },
        "counts": {
            "collections": len(collections),
            "encryption_policies": len(
                encryption_policies
            ),
            "network_policies": len(
                network_policies
            ),
            "data_access_policies": len(
                access_policies
            ),
            "lifecycle_policies": len(
                lifecycle_policies
            ),
        },
        "conflicts": conflicts,
        "preflight_passed": not conflicts,
        "resources_created": False,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print("PREFLIGHT RESULT")
    print("============================================")
    print(
        f"Target exists:  {target_exists}"
    )
    print(
        f"Conflicts:      {len(conflicts)}"
    )
    print(
        "Result:         "
        + (
            "PASSED"
            if not conflicts
            else "FAILED"
        )
    )

    if conflicts:
        for conflict in conflicts:
            print(f"- {conflict}")

    print(
        f"Report:         {OUTPUT_PATH}"
    )
    print("Resources created: False")

    return 0 if not conflicts else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS OpenSearch preflight error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"OpenSearch preflight failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
