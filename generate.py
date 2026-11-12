import json
import numpy as np

DIMENSION = 512

def make_person(seed: int, first: str, last: str, middle: str, filename: str) -> None:
    rng = np.random.default_rng(seed)
    embedding = rng.random(DIMENSION).astype(np.float32)

    # Проверка: норма не должна быть нулевой
    norm = float(np.linalg.norm(embedding))
    assert norm > 0, "Zero embedding"

    payload = {
        "embedding": embedding.tolist(),
        "first_name": first,
        "last_name": last,
        "middle_name": middle,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{filename}: норма = {norm:.4f}, длина = {len(payload['embedding'])}")


make_person(42, "Иван", "Иванов", "Иванович", "person_a.json")
make_person(43, "Пётр", "Петров", "Петрович", "person_b.json")
make_person(999, "Сергей", "Сергеев", "Сергеевич", "person_c.json")