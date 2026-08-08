"""Cross-binding validation for retained canonical training-probe evidence."""
from __future__ import annotations

from scripts.canonical_training_probe_contract import _validate_probe_artifact_base


def _require_equal(field: str, *values: object) -> None:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"PROBE_RUNTIME_CROSS_BINDING_MISMATCH:{field}")


def validate_runtime_cross_binding(
    contract: dict[str, object],
    runtime: dict[str, object],
    hardware_fingerprint: dict[str, object],
) -> None:
    observation = hardware_fingerprint["observed_runtime_and_hardware"]
    hardware_runtime = observation["canonical_runtime"]
    python = observation["python"]
    pytorch = observation["pytorch"]
    environment = observation["execution_environment"]

    _require_equal(
        "canonical_runtime_version",
        contract["canonical_runtime_version"],
        runtime["profile_version"],
        hardware_runtime["profile_version"],
    )
    _require_equal("runtime", runtime, hardware_runtime)

    _require_equal(
        "python_implementation",
        contract["python_implementation"],
        python["implementation"],
    )
    _require_equal("python_version", contract["python_version"], python["version"])
    _require_equal("python_compiler", contract["python_compiler"], python["compiler"])
    _require_equal(
        "python_build",
        contract["python_build"],
        [python["build_number"], python["build_date"]],
    )

    _require_equal("torch_version", contract["torch_version"], pytorch["version"])
    _require_equal(
        "torch_build_configuration_sha256",
        contract["torch_build_configuration_sha256"],
        pytorch["build_configuration_sha256"],
    )
    _require_equal(
        "actual_atten_cpu_capability",
        contract["actual_atten_cpu_capability"],
        pytorch["cpu_dispatch_capability"],
    )
    for field in ("mkl_available", "openmp_available", "mkldnn_available"):
        _require_equal(field, contract[field], pytorch[field])
    _require_equal(
        "mkldnn_enabled",
        contract["mkldnn_enabled"],
        pytorch["mkldnn_enabled"],
        runtime["mkldnn_enabled"],
    )

    _require_equal(
        "ATEN_CPU_CAPABILITY",
        contract["ATEN_CPU_CAPABILITY"],
        environment["ATEN_CPU_CAPABILITY"],
    )
    _require_equal("MKL_CBWR", contract["MKL_CBWR"], environment["MKL_CBWR"])
    for field in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        _require_equal(field, runtime[field], environment[field])


def validate_probe_artifact(payload: object) -> dict[str, object]:
    """Authoritatively validate and cross-bind a sealed retained probe artifact."""
    validated = _validate_probe_artifact_base(payload)
    validate_runtime_cross_binding(
        validated["execution_contract"],
        validated["runtime"],
        validated["hardware_runtime_fingerprint"],
    )
    return validated
