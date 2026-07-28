"""Compatibility wrapper for the v1.18 two-level lock.

New code must import validation.verify_lock directly.
"""
from validation.verify_lock import create as _create, verify as _verify


def create(lock_path, run_id):
    return _create("scientific", lock_path, run_id)


def verify(lock_path):
    return _verify("scientific", lock_path)
