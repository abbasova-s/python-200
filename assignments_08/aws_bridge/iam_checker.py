import json
import fnmatch

def load_policy(path):
    """reads a JSON policy file and returns it as a dictionary"""
    with open(path, "r") as file:
        policy = json.load(file)
    return policy

def statement_matches(statement, action, resource):
    """Check whether a policy statement matches an action and resource."""
    actions = statement["Action"]
    resources = statement["Resource"]

    if isinstance(actions, str):
        actions = [actions]

    if isinstance(resources, str):
        resources = [resources]

    action_matches = any(
        fnmatch.fnmatchcase(action.lower(), pattern.lower())
        for pattern in actions
    )

    resource_matches = any(
        fnmatch.fnmatchcase(resource, pattern)
        for pattern in resources
    )

    return action_matches and resource_matches

def evaluate(policy, action, resource):
    """Evaluate a policy and return its authorization result."""
    matching_effects = [
        statement["Effect"]
        for statement in policy["Statement"]
        if statement_matches(statement, action, resource)
    ]

    if "Deny" in matching_effects:
        return "explicit deny"

    if "Allow" in matching_effects:
        return "allowed"

    return "implicit deny"

def is_allowed(policy, action, resource):
    """Return True only when the policy allows the action."""
    return evaluate(policy, action, resource) == "allowed"


