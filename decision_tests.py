import os
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # корень репозитория
TEST_DIR = BASE_DIR / "test"
RESULTS_DIR = BASE_DIR / "results"


class Question(msgspec.Struct, forbid_unknown_fields=True):
    """Один вопрос запроса /v1/systemone: name → ключ в answers ответа,
    q → instructions, answers → criteria."""

    name: str
    q: str
    # Порядок ключей сохраняется (Python 3.7+ dict ordered): для шкалы уровни
    # идут от низшего к высшему, для да/нет — false/true.
    # choice/noul — map, score — список уровней от низшего к высшему.
    answers: dict[str, str] | list[str]
    type: str = "choice"  # choice | noul | score


class TestSpec(msgspec.Struct, forbid_unknown_fields=True):
    """Тело запроса: состояния и вопросы одного прогона; модель по умолчанию — из $MODEL."""

    states: list[str | dict[str, Any]]  # строка ИЛИ JSON-объект, каждый элемент
    questions: list[Question]
    model: str = os.environ.get("MODEL", "")


class DecisionTestsFile(msgspec.Struct, forbid_unknown_fields=True):
    """Корень файла: один прогон = одна модель + одно состояние."""

    test: TestSpec


class QuestionRequest(msgspec.Struct, forbid_unknown_fields=True):
    """Вопрос в формате /v1/systemone: q → instructions, answers → criteria."""

    instructions: str
    criteria: dict[str, str] | list[str]  # score — список уровней, остальное — map
    type: str = "choice"  # choice | noul | score


class SystemOneRequest(msgspec.Struct, forbid_unknown_fields=True):
    """Тело POST /v1/systemone: name вопроса становится ключом questions."""

    model: str
    state: str | dict[str, Any]
    questions: dict[str, QuestionRequest]

    @classmethod
    def from_test(cls, test: TestSpec) -> Iterator["SystemOneRequest"]:
        """Один вопрос к одному стейту — один запрос /v1/systemone."""
        for state in test.states:
            for q in test.questions:
                yield cls(
                    model=test.model,
                    state=state.strip() if isinstance(state, str) else state,
                    questions={
                        q.name: QuestionRequest(
                            instructions=q.q, criteria=q.answers, type=q.type
                        )
                    },
                )


def format_answer(answer: dict[str, Any]) -> str:
    if answer["type"] == "noul":
        return f"(noul:{answer['noul']:.2f})"
    choice = answer.get("choice")
    weights = " ".join(
        f"{key}={'*' if key == choice else ''}{prob:.2f}"
        for key, prob in answer["probabilities"].items()
    )
    if answer["type"] == "score":
        return f"(score:{answer['score']:.2f} conf:{answer['confidence']:.2f})  {weights}"
    return f"({choice}:{answer['confidence']:.2f})  {weights}"


def round_floats(value: Any) -> Any:
    """Округляет все float в структуре до 4 знаков для записи в YAML."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: round_floats(val) for key, val in value.items()}
    if isinstance(value, list):
        return [round_floats(val) for val in value]
    return value


def run_file(path: Path, host: str) -> Path:
    spec = msgspec.convert(
        yaml.safe_load(path.read_text()), DecisionTestsFile
    ).test

    print(f"model: {spec.model}  file: {path}")

    report: dict[str, list[str]] = {}  # state → [q=..., q=...] в порядке вопросов
    results: list[dict[str, Any]] = [  # снимок прогона для YAML
        {"state": state.strip() if isinstance(state, str) else state,
         "answers": {}}
        for state in spec.states
    ]

    for i, req in enumerate(SystemOneRequest.from_test(spec)):
        http_req = urllib.request.Request(
            f"{host}/v1/systemone",
            data=msgspec.json.encode(req),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_req) as resp:
            answers = msgspec.json.decode(resp.read())["answers"]

        state = req.state
        state_text = state if isinstance(state, str) else msgspec.json.encode(state).decode()

        for name, answer in answers.items():
            results[i // len(spec.questions)]["answers"][name] = round_floats(answer)
            report.setdefault(" ".join(state_text.split()), []).append(
                f"{name}{format_answer(answer)}"
            )

    for state, pairs in report.items():
        print(f"state: {state}")
        for pair in pairs:
            print(f"  {pair}")

    out_path = RESULTS_DIR / f"{spec.model}.{path.name}"
    # путь до источника — от корня репозитория, чтобы было найти исходный тест
    try:
        source = str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source = str(path.resolve())
    out_path.write_text(yaml.safe_dump(
        {"results": {"model": spec.model, "source": source, "states": results}},
        allow_unicode=True,
        sort_keys=False,
    ))
    print(f"results: {out_path}\n")
    return out_path


def main() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    # либо явно перечисленные файлы, либо автоскан всех тестов в test/
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        paths = sorted(TEST_DIR.glob("*.yaml"))

    if not paths:
        print(f"no tests found in {TEST_DIR}")
        return

    for path in paths:
        run_file(path, host)


if __name__ == "__main__":
    main()
