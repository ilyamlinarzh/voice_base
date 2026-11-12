with open("huge_payload.json", "w") as f:
    f.write('{"embedding": [')
    f.write(', '.join(['0.1'] * 1000000)) # 1 миллион чисел
    f.write('], "first_name": "A", "last_name": "B"}')  