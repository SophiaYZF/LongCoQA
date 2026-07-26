from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from longcoqa.paths import PROJECT_ROOT

SETUP = PROJECT_ROOT / "scripts/setup_hipporag2_env.sh"
PATCH = PROJECT_ROOT / "scripts/hipporag2_compat.patch"
FIXTURE_PATCHED_BLOBS = """\
1f5aafbb3168704bc2d0726fb6d88c431be7d463  src/hipporag/embedding_model/__init__.py
3cfb2972c8df8a2e4e70fa3596eae828ec155cb8  src/hipporag/information_extraction/openie_openai.py
57d0c1039d8e85c3394d44e2062c738027d53906  src/hipporag/llm/__init__.py
1553dc3a1fe81498fbabc4c196ab683d3ace742e  src/hipporag/utils/misc_utils.py
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _minimal_hipporag_checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "HippoRAG"
    _write(repo / "src/__init__.py", "")
    _write(repo / "src/hipporag/__init__.py", "")
    _write(repo / "src/hipporag/utils/__init__.py", "")
    _write(
        repo / "src/hipporag/utils/logging_utils.py",
        """
        def get_logger(name):
            return name
        """,
    )
    _write(
        repo / "src/hipporag/utils/config_utils.py",
        """
        class BaseConfig:
            pass
        """,
    )
    _write(
        repo / "src/hipporag/embedding_model/base.py",
        """
        class EmbeddingConfig:
            pass

        class BaseEmbeddingModel:
            pass
        """,
    )
    _write(
        repo / "src/hipporag/embedding_model/__init__.py",
        """
        from .Contriever import ContrieverModel
        from .base import EmbeddingConfig, BaseEmbeddingModel
        from .GritLM import GritLMEmbeddingModel
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
        from .OpenAI import OpenAIEmbeddingModel
        from .Cohere import CohereEmbeddingModel
        from .Transformers import TransformersEmbeddingModel
        from .VLLM import VLLMEmbeddingModel

        from ..utils.logging_utils import get_logger

        logger = get_logger(__name__)


        def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
            if "GritLM" in embedding_model_name:
                return GritLMEmbeddingModel
            elif "NV-Embed-v2" in embedding_model_name:
                return NVEmbedV2EmbeddingModel
            elif "contriever" in embedding_model_name:
                return ContrieverModel
            elif "text-embedding" in embedding_model_name:
                return OpenAIEmbeddingModel
            elif "cohere" in embedding_model_name:
                return CohereEmbeddingModel
            elif embedding_model_name.startswith("Transformers/"):
                return TransformersEmbeddingModel
            elif embedding_model_name.startswith("VLLM/"):
                return VLLMEmbeddingModel
            raise ValueError(f"Unknown embedding model name: {embedding_model_name}")
        """,
    )
    _write(repo / "src/hipporag/llm/base.py", "class BaseLLM:\n    pass\n")
    _write(
        repo / "src/hipporag/llm/openai_gpt.py",
        """
        class CacheOpenAI:
            @classmethod
            def from_experiment_config(cls, config):
                return cls()
        """,
    )
    _write(
        repo / "src/hipporag/llm/__init__.py",
        """
        import os

        from ..utils.logging_utils import get_logger
        from ..utils.config_utils import BaseConfig

        from .openai_gpt import CacheOpenAI
        from .base import BaseLLM
        from .bedrock_llm import BedrockLLM
        from .bedrock_mantle import BedrockMantleLLM
        from .transformers_llm import TransformersLLM


        logger = get_logger(__name__)


        def _get_llm_class(config: BaseConfig):
            if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
                os.environ['OPENAI_API_KEY'] = 'sk-'

            if config.llm_name.startswith('bedrock-mantle/'):
                return BedrockMantleLLM(config)

            if config.llm_name.startswith('bedrock/'):
                return BedrockLLM(config)

            if config.llm_name.startswith('Transformers/'):
                return TransformersLLM(config)

            return CacheOpenAI.from_experiment_config(config)
        """,
    )
    _write(repo / "src/hipporag/information_extraction/__init__.py", "")
    _write(
        repo / "src/hipporag/information_extraction/openie_openai.py",
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def tqdm(values, **kwargs):
            return values

        class OpenIE:
            def ner(self, *args):
                raise AssertionError("empty fixture should not submit NER work")

            def triple_extraction(self, *args):
                raise AssertionError("empty fixture should not submit triple work")

            def run(self, chunks):
                # Extract passages from the provided chunks
                chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

                ner_results_list = []
                total_prompt_tokens = 0
                total_completion_tokens = 0
                num_cache_hit = 0

                with ThreadPoolExecutor() as executor:
                    # Create NER futures for each chunk
                    ner_futures = {
                        executor.submit(self.ner, chunk_key, passage): chunk_key
                        for chunk_key, passage in chunk_passages.items()
                    }

                    pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
                    for future in pbar:
                        result = future.result()
                        ner_results_list.append(result)
                        # Update metrics based on the metadata from the result
                        metadata = result.metadata
                        total_prompt_tokens += metadata.get('prompt_tokens', 0)
                        total_completion_tokens += metadata.get('completion_tokens', 0)
                        if metadata.get('cache_hit'):
                            num_cache_hit += 1

                        pbar.set_postfix({
                            'total_prompt_tokens': total_prompt_tokens,
                            'total_completion_tokens': total_completion_tokens,
                            'num_cache_hit': num_cache_hit
                        })

                triple_results_list = []
                total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
                with ThreadPoolExecutor() as executor:
                    # Create triple extraction futures for each chunk
                    re_futures = {
                        executor.submit(self.triple_extraction, ner_result.chunk_id,
                                        chunk_passages[ner_result.chunk_id],
                                        ner_result.unique_entities): ner_result.chunk_id
                        for ner_result in ner_results_list
                    }
                    # Collect triple extraction results with progress bar
                    pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
                    for future in pbar:
                        result = future.result()
                        triple_results_list.append(result)
                        metadata = result.metadata
                        total_prompt_tokens += metadata.get('prompt_tokens', 0)
                        total_completion_tokens += metadata.get('completion_tokens', 0)
                        if metadata.get('cache_hit'):
                            num_cache_hit += 1
                        pbar.set_postfix({
                            'total_prompt_tokens': total_prompt_tokens,
                            'total_completion_tokens': total_completion_tokens,
                            'num_cache_hit': num_cache_hit
                        })
                return ner_results_list, triple_results_list
        """,
    )
    misc_prefix = textwrap.dedent(
        """
        from __future__ import annotations
        import re
        from typing import Dict

        class NerRawOutput:
            pass

        class TripleRawOutput:
            pass
        """
    ).lstrip()
    # Keep the target function near its upstream line number so git-apply's
    # bounded offset search exercises the real hunk rather than a rewritten one.
    misc_body = textwrap.dedent(
        """

        def text_processing(text):
            if isinstance(text, list):
                return [text_processing(t) for t in text]
            if not isinstance(text, str):
                text = str(text)
            return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

        def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

            return {}, {}
        """
    )
    _write(
        repo / "src/hipporag/utils/misc_utils.py",
        misc_prefix + ("\n" * 70) + misc_body,
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "anonymous@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture base"], cwd=repo, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, revision


def _fixture_state_file(repo: Path) -> Path:
    state_file = repo.parent / "hipporag2_compat.blobs"
    state_file.write_text(FIXTURE_PATCHED_BLOBS, encoding="utf-8")
    return state_file


def _apply_patch(repo: Path, revision: str):
    return _run(
        [
            "bash",
            "-c",
            'source "$1"; longcoqa_apply_hipporag2_compat "$2" "$3" "$4" "$5"',
            "bash",
            str(SETUP),
            str(repo),
            revision,
            str(PATCH),
            str(_fixture_state_file(repo)),
        ]
    )


def _run_setup_main(repo: Path, revision: str, env_path: Path):
    return _run(
        [
            "bash",
            "-c",
            'source "$1"; '
            'HIPPO_ROOT="$2"; HIPPO_REVISION="$3"; HIPPO_PATCH="$4"; '
            'HIPPO_STATE="$5"; ENV="$6"; BASE_PYTHON="$7"; main',
            "bash",
            str(SETUP),
            str(repo),
            revision,
            str(PATCH),
            str(_fixture_state_file(repo)),
            str(env_path),
            str(env_path.parent / "python-must-not-run"),
        ]
    )


def test_compatibility_patch_applies_and_is_idempotent(tmp_path: Path) -> None:
    repo, revision = _minimal_hipporag_checkout(tmp_path)

    first = _apply_patch(repo, revision)
    assert first.returncode == 0, first.stderr
    assert "HippoRAG compatibility patch applied" in first.stdout
    first_diff = subprocess.check_output(["git", "diff"], cwd=repo, text=True)

    second = _apply_patch(repo, revision)
    assert second.returncode == 0, second.stderr
    assert "already applied" in second.stdout
    assert subprocess.check_output(["git", "diff"], cwd=repo, text=True) == first_diff

    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=repo, text=True
    ).splitlines()
    assert changed == [
        "src/hipporag/embedding_model/__init__.py",
        "src/hipporag/information_extraction/openie_openai.py",
        "src/hipporag/llm/__init__.py",
        "src/hipporag/utils/misc_utils.py",
    ]

    behavior = _run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import src.hipporag.embedding_model
                import src.hipporag.llm
                from src.hipporag.information_extraction import openie_openai
                from src.hipporag.utils.misc_utils import text_processing

                assert text_processing("哈利·波特") == "哈利 波特"
                workers = []
                class Executor:
                    def __init__(self, max_workers=None):
                        workers.append(max_workers)
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        return False
                    def submit(self, *args):
                        raise AssertionError("empty fixture should submit no work")
                openie_openai.ThreadPoolExecutor = Executor
                openie_openai.OpenIE().run({})
                assert workers == [4, 4]
                """
            ),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    assert behavior.returncode == 0, behavior.stderr


@pytest.mark.parametrize(
    "state",
    ("extra-target", "extra-nontarget", "staged", "partial"),
)
def test_setup_rejects_noncanonical_tracked_state_before_installing(
    tmp_path: Path, state: str
) -> None:
    repo, revision = _minimal_hipporag_checkout(tmp_path)
    if state == "partial":
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--include=src/hipporag/utils/misc_utils.py",
                str(PATCH),
            ],
            cwd=repo,
            check=True,
        )
    else:
        applied = _apply_patch(repo, revision)
        assert applied.returncode == 0, applied.stderr
        if state == "extra-target":
            with (repo / "src/hipporag/utils/misc_utils.py").open("a", encoding="utf-8") as handle:
                handle.write("\n# unexpected target-file edit\n")
        elif state == "extra-nontarget":
            with (repo / "src/hipporag/utils/config_utils.py").open("a", encoding="utf-8") as handle:
                handle.write("\n# unexpected non-target edit\n")
        else:
            subprocess.run(
                ["git", "add", "src/hipporag/utils/misc_utils.py"],
                cwd=repo,
                check=True,
            )

    env_path = tmp_path / "must-not-be-created"
    result = _run_setup_main(repo, revision, env_path)

    assert result.returncode != 0
    assert "Refusing HippoRAG tracked state" in result.stderr
    assert not env_path.exists()


def test_setup_rejects_checkout_at_wrong_revision_before_installing(tmp_path: Path) -> None:
    repo, _revision = _minimal_hipporag_checkout(tmp_path)
    env = {
        **os.environ,
        "HIPPO_ROOT": str(repo),
        "HIPPO_ENV": str(tmp_path / "must-not-be-created"),
        "LONGCOQA_PYTHON": "/bin/false",
    }

    result = _run(["bash", str(SETUP)], env=env)

    assert result.returncode != 0
    assert "Expected HippoRAG revision 1e8f60981bf760b64003aa5bf5668126d0c106b3" in result.stderr
    assert not (tmp_path / "must-not-be-created").exists()
