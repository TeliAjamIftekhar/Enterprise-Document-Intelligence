from __future__ import annotations

import argparse
import hashlib
import json
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


REGION = "us-east-1"
ACCOUNT_ID = "334590195171"

ROLE_NAME = "genaiapp"
ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"
)

GROUP_NAME = "edi-textbook-nextgen"
COLLECTION_NAME = "edi-textbook-vector"
COLLECTION_TYPE = "VECTORSEARCH"

ENCRYPTION_POLICY_NAME = "edi-textbook-encryption"
NETWORK_POLICY_NAME = "edi-textbook-network"
ACCESS_POLICY_NAME = "edi-textbook-data"

OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "provisioning-report.json"
)

CAPACITY_LIMITS = {
    "maxIndexingCapacityInOCU": 2.0,
    "maxSearchCapacityInOCU": 2.0,
    "minIndexingCapacityInOCU": 0.0,
    "minSearchCapacityInOCU": 0.0,
}

TAGS = [
    {
        "key": "Project",
        "value": "EnterpriseDocumentIntelligence",
    },
    {
        "key": "Environment",
        "value": "Development",
    },
    {
        "key": "Book",
        "value": "grade-9-english-kaveri",
    },
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def deterministic_token(
    resource_type: str,
    resource_name: str,
) -> str:
    raw_value = (
        f"{ACCOUNT_ID}|{REGION}|"
        f"{resource_type}|{resource_name}"
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()

    return (
        f"edi-{resource_type}-"
        f"{digest[:48]}"
    )


def clean_response(
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.items()
        if key != "ResponseMetadata"
    }


def parse_policy(
    value: Any,
) -> Any:
    if isinstance(value, str):
        return json.loads(value)

    return value


def list_all(
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
    records: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()

    while True:
        response = operation(**request)

        values = response.get(
            result_key,
            [],
        )

        if isinstance(values, list):
            records.extend(values)

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
                f"Repeated nextToken from "
                f"{operation_name}."
            )

        seen_tokens.add(next_token)
        request["nextToken"] = next_token

    return records


def get_security_policy(
    client: Any,
    policy_type: str,
    name: str,
) -> dict[str, Any] | None:
    summaries = list_all(
        client=client,
        operation_name="list_security_policies",
        result_key="securityPolicySummaries",
        type=policy_type,
    )

    if not any(
        record.get("name") == name
        for record in summaries
    ):
        return None

    response = client.get_security_policy(
        type=policy_type,
        name=name,
    )

    detail = response.get(
        "securityPolicyDetail"
    )

    if not isinstance(detail, dict):
        raise RuntimeError(
            f"Security policy detail missing: {name}"
        )

    return detail


def get_access_policy(
    client: Any,
    name: str,
) -> dict[str, Any] | None:
    summaries = list_all(
        client=client,
        operation_name="list_access_policies",
        result_key="accessPolicySummaries",
        type="data",
    )

    if not any(
        record.get("name") == name
        for record in summaries
    ):
        return None

    response = client.get_access_policy(
        type="data",
        name=name,
    )

    detail = response.get(
        "accessPolicyDetail"
    )

    if not isinstance(detail, dict):
        raise RuntimeError(
            f"Data access policy detail missing: {name}"
        )

    return detail


def ensure_security_policy(
    client: Any,
    policy_type: str,
    name: str,
    description: str,
    expected_policy: Any,
    create: bool,
) -> dict[str, Any]:
    existing = get_security_policy(
        client=client,
        policy_type=policy_type,
        name=name,
    )

    if existing is not None:
        actual_policy = parse_policy(
            existing.get("policy")
        )

        if (
            canonical_json(actual_policy)
            != canonical_json(expected_policy)
        ):
            raise RuntimeError(
                f"Existing {policy_type} policy "
                f"does not match expected policy: {name}"
            )

        print(
            f"MATCHING {policy_type} policy: {name}"
        )

        return {
            "name": name,
            "type": policy_type,
            "action": "matching",
            "detail": existing,
        }

    if not create:
        print(
            f"PLANNED  {policy_type} policy: {name}"
        )

        return {
            "name": name,
            "type": policy_type,
            "action": "planned",
        }

    response = client.create_security_policy(
        name=name,
        type=policy_type,
        description=description,
        policy=canonical_json(
            expected_policy
        ),
        clientToken=deterministic_token(
            policy_type,
            name,
        ),
    )

    print(
        f"CREATED  {policy_type} policy: {name}"
    )

    return {
        "name": name,
        "type": policy_type,
        "action": "created",
        "detail": clean_response(response),
    }


def ensure_access_policy(
    client: Any,
    expected_policy: Any,
    create: bool,
) -> dict[str, Any]:
    existing = get_access_policy(
        client=client,
        name=ACCESS_POLICY_NAME,
    )

    if existing is not None:
        actual_policy = parse_policy(
            existing.get("policy")
        )

        policies_match = (
            canonical_json(actual_policy)
            == canonical_json(expected_policy)
        )

        if policies_match:
            print(
                f"MATCHING data policy: "
                f"{ACCESS_POLICY_NAME}"
            )

            return {
                "name": ACCESS_POLICY_NAME,
                "type": "data",
                "action": "matching",
                "detail": existing,
            }

        if not create:
            print(
                f"PLANNED  data policy update: "
                f"{ACCESS_POLICY_NAME}"
            )

            return {
                "name": ACCESS_POLICY_NAME,
                "type": "data",
                "action": "update-planned",
                "current_detail": existing,
            }

        policy_version = existing.get(
            "policyVersion"
        )

        if not isinstance(
            policy_version,
            str,
        ) or not policy_version:
            raise RuntimeError(
                "Existing data policy has no "
                "policyVersion."
            )

        response = client.update_access_policy(
            name=ACCESS_POLICY_NAME,
            type="data",
            policyVersion=policy_version,
            description=(
                "Least-privilege data access for the "
                "EDI textbook vector collection."
            ),
            policy=canonical_json(
                expected_policy
            ),
            clientToken=deterministic_token(
                "data-update-least-privilege",
                ACCESS_POLICY_NAME,
            ),
        )

        updated_detail = response.get(
            "accessPolicyDetail",
            {},
        )

        updated_policy = parse_policy(
            updated_detail.get("policy")
        )

        if (
            canonical_json(updated_policy)
            != canonical_json(expected_policy)
        ):
            raise RuntimeError(
                "Updated data access policy does not "
                "match the expected policy."
            )

        print(
            f"UPDATED  data policy: "
            f"{ACCESS_POLICY_NAME}"
        )

        return {
            "name": ACCESS_POLICY_NAME,
            "type": "data",
            "action": "updated",
            "detail": updated_detail,
        }

    if not create:
        print(
            f"PLANNED  data policy: "
            f"{ACCESS_POLICY_NAME}"
        )

        return {
            "name": ACCESS_POLICY_NAME,
            "type": "data",
            "action": "planned",
        }

    response = client.create_access_policy(
        name=ACCESS_POLICY_NAME,
        type="data",
        description=(
            "Least-privilege data access for the "
            "EDI textbook vector collection."
        ),
        policy=canonical_json(
            expected_policy
        ),
        clientToken=deterministic_token(
            "data",
            ACCESS_POLICY_NAME,
        ),
    )

    print(
        f"CREATED  data policy: "
        f"{ACCESS_POLICY_NAME}"
    )

    return {
        "name": ACCESS_POLICY_NAME,
        "type": "data",
        "action": "created",
        "detail": clean_response(response),
    }


def get_collection_group(
    client: Any,
) -> dict[str, Any] | None:
    summaries = list_all(
        client=client,
        operation_name="list_collection_groups",
        result_key="collectionGroupSummaries",
    )

    if not any(
        record.get("name") == GROUP_NAME
        for record in summaries
    ):
        return None

    response = client.batch_get_collection_group(
        names=[GROUP_NAME]
    )

    details = response.get(
        "collectionGroupDetails",
        [],
    )

    if not isinstance(
        details,
        list,
    ) or len(details) != 1:
        raise RuntimeError(
            "Could not resolve collection group detail."
        )

    return details[0]


def validate_collection_group(
    detail: dict[str, Any],
) -> None:
    if detail.get("generation") != "NEXTGEN":
        raise RuntimeError(
            "Existing collection group is not NEXTGEN."
        )

    if detail.get(
        "standbyReplicas"
    ) != "ENABLED":
        raise RuntimeError(
            "Existing collection group standby "
            "configuration differs."
        )

    actual_capacity = detail.get(
        "capacityLimits",
        {},
    )

    for field, expected in (
        CAPACITY_LIMITS.items()
    ):
        actual = actual_capacity.get(field)

        if float(actual) != float(expected):
            raise RuntimeError(
                "Existing collection group capacity "
                f"differs for {field}: "
                f"expected={expected}, actual={actual}"
            )


def ensure_collection_group(
    client: Any,
    create: bool,
) -> dict[str, Any]:
    existing = get_collection_group(client)

    if existing is not None:
        validate_collection_group(existing)

        print(
            f"MATCHING collection group: {GROUP_NAME}"
        )

        return {
            "name": GROUP_NAME,
            "action": "matching",
            "detail": existing,
        }

    if not create:
        print(
            f"PLANNED  collection group: {GROUP_NAME}"
        )

        return {
            "name": GROUP_NAME,
            "action": "planned",
        }

    response = client.create_collection_group(
        name=GROUP_NAME,
        description=(
            "NextGen scale-to-zero group for "
            "textbook retrieval collections."
        ),
        generation="NEXTGEN",
        standbyReplicas="ENABLED",
        capacityLimits=CAPACITY_LIMITS,
        tags=TAGS,
        clientToken=deterministic_token(
            "group-nextgen-enabled",
            GROUP_NAME,
        ),
    )

    print(
        f"CREATED  collection group: {GROUP_NAME}"
    )

    detail = response.get(
        "createCollectionGroupDetail",
        {},
    )

    return {
        "name": GROUP_NAME,
        "action": "created",
        "detail": detail,
    }


def get_collection(
    client: Any,
) -> dict[str, Any] | None:
    response = client.batch_get_collection(
        names=[COLLECTION_NAME]
    )

    details = response.get(
        "collectionDetails",
        [],
    )

    if not details:
        return None

    if not isinstance(
        details,
        list,
    ) or len(details) != 1:
        raise RuntimeError(
            "Unexpected collection detail count."
        )

    return details[0]


def validate_collection(
    detail: dict[str, Any],
) -> None:
    if detail.get("type") != COLLECTION_TYPE:
        raise RuntimeError(
            "Existing collection has an incompatible "
            f"type: {detail.get('type')}"
        )

    if detail.get(
        "collectionGroupName"
    ) != GROUP_NAME:
        raise RuntimeError(
            "Existing collection belongs to another "
            "collection group."
        )

    if detail.get(
        "standbyReplicas"
    ) != "ENABLED":
        raise RuntimeError(
            "Existing collection standby setting differs."
        )

    if detail.get(
        "deletionProtection"
    ) != "ENABLED":
        raise RuntimeError(
            "Existing collection deletion protection "
            "is not enabled."
        )


def ensure_collection(
    client: Any,
    create: bool,
) -> dict[str, Any]:
    existing = get_collection(client)

    if existing is not None:
        validate_collection(existing)

        print(
            f"MATCHING collection: {COLLECTION_NAME}"
        )

        return {
            "name": COLLECTION_NAME,
            "action": "matching",
            "detail": existing,
        }

    if not create:
        print(
            f"PLANNED  collection: {COLLECTION_NAME}"
        )

        return {
            "name": COLLECTION_NAME,
            "action": "planned",
        }

    response = client.create_collection(
        name=COLLECTION_NAME,
        type=COLLECTION_TYPE,
        collectionGroupName=GROUP_NAME,
        description=(
            "Vector and lexical retrieval collection "
            "for versioned textbook indexes."
        ),
        standbyReplicas="ENABLED",
        deletionProtection="ENABLED",
        tags=TAGS,
        clientToken=deterministic_token(
            "collection-nextgen-enabled",
            COLLECTION_NAME,
        ),
    )

    print(
        f"CREATED  collection: {COLLECTION_NAME}"
    )

    return {
        "name": COLLECTION_NAME,
        "action": "created",
        "detail": clean_response(response),
    }


def wait_for_collection(
    client: Any,
    timeout_seconds: int = 900,
    interval_seconds: int = 10,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0

    while True:
        attempt += 1

        detail = get_collection(client)

        if detail is None:
            status = "NOT_FOUND"
        else:
            status = str(
                detail.get("status", "UNKNOWN")
            )

        print(
            f"Collection status check "
            f"{attempt}: {status}"
        )

        if status == "ACTIVE":
            validate_collection(detail)
            return detail

        if status in {
            "FAILED",
            "UPDATE_FAILED",
        }:
            raise RuntimeError(
                "Collection entered failure state:\n"
                + json.dumps(
                    detail,
                    indent=2,
                    default=str,
                )
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out while waiting for collection "
                "to become ACTIVE."
            )

        time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Provision the EDI OpenSearch Serverless "
            "NextGen vector collection."
        )
    )

    parser.add_argument(
        "--create",
        action="store_true",
        help=(
            "Create missing resources. Without this "
            "option, only a safe plan is generated."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sts = boto3.client(
        "sts",
        region_name=REGION,
    )

    client = boto3.client(
        "opensearchserverless",
        region_name=REGION,
    )

    identity = clean_response(
        sts.get_caller_identity()
    )

    if identity.get("Account") != ACCOUNT_ID:
        raise RuntimeError(
            "Unexpected AWS account."
        )

    encryption_policy = {
        "Rules": [
            {
                "ResourceType": "collection",
                "Resource": [
                    f"collection/{COLLECTION_NAME}"
                ],
            }
        ],
        "AWSOwnedKey": True,
    }

    network_policy = [
        {
            "Description": (
                "Public endpoint access for the EDI "
                "development collection. IAM and the "
                "data access policy remain required."
            ),
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [
                        f"collection/{COLLECTION_NAME}"
                    ],
                },
                {
                    "ResourceType": "dashboard",
                    "Resource": [
                        f"collection/{COLLECTION_NAME}"
                    ],
                },
            ],
            "AllowFromPublic": True,
        }
    ]

    data_access_policy = [
        {
            "Description": (
                "Full development access for the "
                "genaiapp execution role."
            ),
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [
                        f"collection/{COLLECTION_NAME}"
                    ],
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:UpdateCollectionItems",
                        "aoss:DescribeCollectionItems",
                    ],
                },
                {
                    "ResourceType": "index",
                    "Resource": [
                        f"index/{COLLECTION_NAME}/*"
                    ],
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:UpdateIndex",
                        "aoss:DescribeIndex",
                        "aoss:ReadDocument",
                        "aoss:WriteDocument",
                    ],
                },
            ],
            "Principal": [
                ROLE_ARN
            ],
        }
    ]

    print("============================================")
    print("OPENSEARCH NEXTGEN PROVISIONING")
    print("============================================")
    print(f"Region:             {REGION}")
    print(f"Account:            {ACCOUNT_ID}")
    print(f"Caller:             {identity.get('Arn')}")
    print(f"Principal:          {ROLE_ARN}")
    print(f"Collection group:   {GROUP_NAME}")
    print(f"Collection:         {COLLECTION_NAME}")
    print(f"Collection type:    {COLLECTION_TYPE}")
    print(f"Create:             {args.create}")
    print()
    print("Capacity limits:")
    print(
        json.dumps(
            CAPACITY_LIMITS,
            indent=2,
        )
    )
    print()

    resources = []

    resources.append(
        ensure_security_policy(
            client=client,
            policy_type="encryption",
            name=ENCRYPTION_POLICY_NAME,
            description=(
                "AWS-owned encryption key policy for "
                "the EDI textbook vector collection."
            ),
            expected_policy=encryption_policy,
            create=args.create,
        )
    )

    resources.append(
        ensure_security_policy(
            client=client,
            policy_type="network",
            name=NETWORK_POLICY_NAME,
            description=(
                "Development network access policy for "
                "the EDI textbook vector collection."
            ),
            expected_policy=network_policy,
            create=args.create,
        )
    )

    resources.append(
        ensure_access_policy(
            client=client,
            expected_policy=data_access_policy,
            create=args.create,
        )
    )

    resources.append(
        ensure_collection_group(
            client=client,
            create=args.create,
        )
    )

    resources.append(
        ensure_collection(
            client=client,
            create=args.create,
        )
    )

    final_collection = get_collection(client)

    if args.create:
        final_collection = wait_for_collection(
            client=client,
        )

    status = (
        "PROVISIONED"
        if args.create
        else "PLANNED"
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": status,
        "region": REGION,
        "identity": identity,
        "principal_arn": ROLE_ARN,
        "configuration": {
            "collection_group_name": GROUP_NAME,
            "collection_name": COLLECTION_NAME,
            "collection_type": COLLECTION_TYPE,
            "generation": "NEXTGEN",
            "capacity_limits": CAPACITY_LIMITS,
            "standby_replicas": "ENABLED",
            "deletion_protection": "ENABLED",
            "network_access": "PUBLIC",
            "encryption": "AWS_OWNED_KEY",
        },
        "resources": resources,
        "collection_detail": final_collection,
        "resources_created": args.create,
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
    print("PROVISIONING RESULT")
    print("============================================")
    print(f"Status:     {status}")

    if final_collection:
        print(
            f"Collection: "
            f"{final_collection.get('name')}"
        )
        print(
            f"State:      "
            f"{final_collection.get('status')}"
        )
        print(
            f"ID:         "
            f"{final_collection.get('id')}"
        )
        print(
            f"Endpoint:   "
            f"{final_collection.get('collectionEndpoint')}"
        )
    else:
        print(
            "Collection: not created during preflight"
        )

    print(f"Report:     {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS provisioning error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Provisioning failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
