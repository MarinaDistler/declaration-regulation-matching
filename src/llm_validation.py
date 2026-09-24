import json
from pathlib import Path

def make_llm_prompt(
    declaration_id,
    declaration_text,
    candidates,
):
    candidate_text = "\n\n".join(
        f"regulation_id: {candidate['regulation_id']}\n"
        f"npa:\n{candidate['npa']}"
        for candidate in candidates
    )

    return f"""Ты выполняешь роль эксперта по сопоставлению описаний товаров
с текстами нормативно-правовых актов (НПА).

Для каждого кандидата оцени, есть ли содержательная связь между
описанием товара в декларации и текстом НПА.

Шкала оценки:

1 — НПА содержательно связан с товаром: он относится к тому же товару,
типу продукции, классу продукции, назначению или явно указанным
характеристикам товара. Не требуется, чтобы из предоставленных текстов
можно было доказать прямую юридическую применимость НПА.

0 — содержательной связи между товаром и НПА по предоставленным текстам
не видно.

Важно:
- Оценивай только на основании текста декларации и текста НПА.
- Не используй внешние знания.
- Не придумывай отсутствующие характеристики товара.
- Не требуй доказательства юридической применимости НПА.
- Если НПА относится к тому же типу товара или к явно связанным
  характеристикам, это может быть оценено как 1, даже если точное
  соответствие или юридическая применимость не установлены.
- Не ставь 1 только из-за отдельных совпадающих слов, если содержательной
  связи между товарами нет.

Верни ровно один результат для каждого кандидата.
Не пропускай кандидатов и не добавляй новые.
Сохрани regulation_id без изменений.

Верни только JSON без markdown и пояснений.

ДЕКЛАРАЦИЯ
declaration_id: {declaration_id}

text:
{declaration_text}

КАНДИДАТЫ НПА

{candidate_text}

Формат ответа:

{{
  "declaration_id": "{declaration_id}",
  "results": [
    {{"regulation_id": "NPA0001", "relevance": 1}},
    {{"regulation_id": "NPA0002", "relevance": 0}}
  ]
}}
"""


def validate_llm_answer(answer, candidate_ids):
    if isinstance(answer, str):
        answer = json.loads(answer)

    if "declaration_id" not in answer:
        raise ValueError("Missing declaration_id")

    results = answer.get("results")

    if not isinstance(results, list):
        raise ValueError("results must be a list")

    actual_ids = [
        item.get("regulation_id")
        for item in results
    ]

    if len(actual_ids) != len(candidate_ids):
        raise ValueError(
            f"Expected {len(candidate_ids)} results, "
            f"got {len(actual_ids)}"
        )

    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Duplicate regulation_id")

    if set(actual_ids) != set(candidate_ids):
        raise ValueError("Candidate IDs do not match")

    for item in results:
        if item.get("relevance") not in {0, 1}:
            raise ValueError(
                f"Invalid relevance: {item}"
            )

    return answer


def save_llm_answer(answer, path):
    path = Path(path)

    existing_ids = set()

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing = json.loads(line)
                    existing_ids.add(existing["declaration_id"])

    declaration_id = answer["declaration_id"]

    if declaration_id in existing_ids:
        raise ValueError(
            f"Answer for declaration {declaration_id} already exists"
        )

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(answer, ensure_ascii=False) + "\n"
        )

def load_llm_answers(path):
    answers = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            answer = json.loads(line)

            declaration_id = str(answer["declaration_id"])

            answers[declaration_id] = {
                str(item["regulation_id"]): int(item["relevance"])
                for item in answer["results"]
            }

    return answers