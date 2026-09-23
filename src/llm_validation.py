import json


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
с регулирующими нормативно-правовыми актами (НПА).

Для каждого НПА оцени релевантность по шкале:

2 — НПА непосредственно регулирует данный товар, его тип,
характеристики или условия его обращения.

1 — НПА тематически связан с товаром, но недостаточно
оснований считать его непосредственно применимым.

0 — НПА не относится к товару.

Оценивай только на основании предоставленных текстов.
Не используй внешние знания и не придумывай отсутствующие характеристики.

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
    {{"regulation_id": "NPA0001", "relevance": 2}},
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
        if item.get("relevance") not in {0, 1, 2}:
            raise ValueError(
                f"Invalid relevance: {item}"
            )

    return answer


def save_llm_answer(answer, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(answer, ensure_ascii=False)
            + "\n"
        )