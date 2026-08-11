import ast
from pathlib import Path


def test_import_architecture_isolation() -> None:
    """Enforce strict architecture rule: Only LiteLLMClient imports litellm."""
    src_dir = Path(__file__).parent.parent.parent / "src" / "personal_ai"
    litellm_client_path = (src_dir / "llm" / "litellm_client.py").resolve()

    litellm_imports = []

    for py_file in src_dir.rglob("*.py"):
        resolved_path = py_file.resolve()
        with open(resolved_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "litellm" or alias.name.startswith("litellm."):
                        litellm_imports.append(resolved_path)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "litellm" or (node.module and node.module.startswith("litellm.")):
                    litellm_imports.append(resolved_path)

    # Only litellm_client.py is allowed to import litellm
    allowed_imports = {litellm_client_path}
    actual_imports = set(litellm_imports)

    illegal_imports = actual_imports - allowed_imports
    assert not illegal_imports, f"Illegal LiteLLM imports found in: {illegal_imports}"
