"""Level 03: decode with bounded context window."""


def truncate_prefix(prefix, max_context):
    if prefix.shape[1] <= max_context:
        return prefix
    return prefix[:, -max_context:, :]
