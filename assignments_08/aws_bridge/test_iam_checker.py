from pathlib import Path

import pytest

from iam_checker import evaluate, load_policy, is_allowed, statement_matches


POLICY_PATH = Path(__file__).parent / "submission_alert_policy.json"
POLICY = load_policy(POLICY_PATH)

@pytest.mark.parametrize(
    "action, resource, expected",
    [
        (
            "ssm:GetParameter",
            "arn:aws:ssm:us-east-1:123456789012:parameter/ctd/submission-alert/threshold",
            "allowed",
        ),
        (
            "SSM:getparameter",
            "arn:aws:ssm:us-east-1:123456789012:parameter/ctd/submission-alert/threshold",
            "allowed",
        ),
        (
            "ssm:GetParameter",
            "arn:aws:ssm:us-east-1:123456789012:parameter/ctd/submission-alert/admin/override",
            "explicit deny",
        ),
        (
            "ssm:PutParameter",
            "arn:aws:ssm:us-east-1:123456789012:parameter/ctd/submission-alert/threshold",
            "implicit deny",
        ),
        (
            "cloudwatch:PutMetricData",
            "*",
            "allowed",
        ),
        (
            "s3:GetObject",
            "arn:aws:s3:::ctd-gradebook/week07.csv",
            "implicit deny",
        ),
    ],
)
def test_evaluate_policy(action, resource, expected):
    assert evaluate(POLICY, action, resource) == expected


def test_wildcard_action():
    statement = {
        "Effect": "Allow",
        "Action": "s3:Get*",
        "Resource": "arn:aws:s3:::my-bucket/*",
    }

    assert statement_matches(
        statement,
        "s3:GetObject",
        "arn:aws:s3:::my-bucket/file.csv",
    )


def test_resource_matching_is_case_sensitive():
    statement = {
        "Effect": "Allow",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-bucket/file.csv",
    }

    assert not statement_matches(
        statement,
        "s3:GetObject",
        "arn:aws:s3:::my-bucket/FILE.csv",
    )