"""
Общий код проекта LinguaBot: данные, промпты, вызов модели.

"""

import json, re, time

BASE = "/content/drive/MyDrive/linguaBot"
RULES = RULE_BY_ID = SCEN = None
_model = _tok = None

def init(base=BASE):
    """Загрузить каталог правил и сценарии из папки проекта."""
    global BASE, RULES, RULE_BY_ID, SCEN
    BASE = base
    load = lambda name: json.load(open(f"{BASE}/{name}", encoding="utf-8"))
    RULES = load("grammar_rules.json")
    RULE_BY_ID = {r["rule_id"]: r for r in RULES}
    SCEN = {s["scenario_id"]: s for s in load("scenarios.json")}
    # Проверка совместимости: каталог и сценарии должны ссылаться на одни и те же rule_id
    for s in SCEN.values():
        for r in s["target_rules"]:
            if r not in RULE_BY_ID:
                raise ValueError(f'{s["scenario_id"]}: rule_id "{r}" нет в grammar_rules.json')
            if s["level"] not in RULE_BY_ID[r]["levels"]:
                print(f'{s["scenario_id"]}: правило {r} не доступно на уровне {s["level"]}')
    for b, txt in FEWSHOT.items():
        for r in re.findall(r'"rule_id": "([^"]+)"', txt):
            if r not in RULE_BY_ID:
                raise ValueError(f'FEWSHOT {b}: rule_id "{r}" нет в каталоге')
    print(f"Каталог: {len(RULES)} правил, сценариев: {len(SCEN)}")

def setup(model, tok):
    """Передать загруженную модель и токенизатор"""
    global _model, _tok
    _model, _tok = model, tok

def topic_for_turn(turn_no, n_topics):
    """Текущая подтема по номеру реплики ученика (4 реплики на подтему; в приложении это считает backend)."""
    return min((turn_no - 1) // 4, n_topics - 1)

#  ПРОМПТЫ 
# В обучающих данных system-промпт должен совпадать с этим и с тем, что будет в приложении.
def band(level):
    return "A1-A2" if level in ("A1", "A2") else "B1-B2"

# Структура реплики бота: реакция на ученика + короткая фраза о себе / о происходящем (образец целевой грамматики) + открытый вопрос.
# A1-A2 — до 3 очень коротких предложений, B1-B2 — 2-4.

STYLE = {
    "A1-A2": "Use simple, frequent words plus the target vocabulary. React to the student and/or add a short sentence about yourself or the situation (use the target grammar). Vary the length: 2-3 very short sentences, up to 8 words each.",
    "B1-B2": "Use the target vocabulary and some synonyms. React to the student and/or share a short opinion or experience (use the target grammar). Vary the length: 2-4 sentences."
}

STYLE_FREE = {
    "A1-A2": "Use simple, frequent words. React to the student and/or add a short sentence about yourself. Vary the length: 2-3 very short sentences, up to 8 words each.",
    "B1-B2": "Use varied vocabulary and some synonyms. React to the student and/or share a short opinion or experience. Vary the length: 2-4 sentences."
}

POLICY = {
    "A1-A2": "Correct all language mistakes (grammar, vocabulary, spelling). Never correct facts, opinions, or what the student chose to say, and do not correct short or one-word answers that are acceptable in conversation.",
    "B1-B2": "Correct all language mistakes (grammar, vocabulary, spelling). Never correct facts, opinions, or what the student chose to say, and do not correct short or one-word answers that are acceptable in conversation."
}

NON_EXPLAINABLE = {"word_choice", "spelling", "dependent_preposition", "other"}

USE_FEWSHOT = True       # True — добавить в промпт примеры полного ответа
FEWSHOT = {
    "A1-A2": """
Examples of full answers. The whole student reply goes into "corrected" (copy of the input if there are no mistakes). Each correction has exactly one rule_id. List ALL mistakes in the order they appear. Never correct facts, opinions, or short acceptable answers.
Student: "Yesterday I go to museum and made a lot of photos." -> {"corrected": "Yesterday I went to the museum and took a lot of photos.", "corrections": [{"original": "I go", "correct": "I went", "rule_id": "past_tense_needed"}, {"original": "to museum", "correct": "to the museum", "rule_id": "article_the"}, {"original": "made a lot of photos", "correct": "took a lot of photos", "rule_id": "word_choice"}], "reply": "Sounds fun! I went to a museum last week too. What did you like most there?"}
Student: "My sister like music." -> {"corrected": "My sister likes music.", "corrections": [{"original": "My sister like", "correct": "My sister likes", "rule_id": "present_simple_s"}], "reply": "Nice! I like music too, especially jazz. What music does your sister like?"}
Student: "I live in a small town." -> {"corrected": "I live in a small town.", "corrections": [], "reply": "Cool! I live in a small town too. What do you like about your town?"}
Student: "I saw tigers in the park." -> {"corrected": "I saw tigers in the park.", "corrections": [], "reply": "Wow, tigers! That sounds exciting. What else did you see?"}""",
    "B1-B2": """
Examples of full answers. The whole student reply goes into "corrected" (copy of the input if there are no mistakes). Each correction has exactly one rule_id. List ALL mistakes in the order they appear. Never correct facts, opinions, or short acceptable answers.
Student: "I am wanting to visit Paris because I have been there in 2019 and I liked very much it." -> {"corrected": "I want to visit Paris because I was there in 2019 and I liked it very much.", "corrections": [{"original": "I am wanting", "correct": "I want", "rule_id": "stative_continuous"}, {"original": "I have been there in 2019", "correct": "I was there in 2019", "rule_id": "present_perfect_vs_past"}, {"original": "liked very much it", "correct": "liked it very much", "rule_id": "word_order"}], "reply": "Paris is beautiful in spring. What would you like to see first?"}
Student: "He must to leave early." -> {"corrected": "He must leave early.", "corrections": [{"original": "must to leave", "correct": "must leave", "rule_id": "modal_base_verb"}], "reply": "That is a pity, the party is only starting. Why does he have to go?"}
Student: "If I had more time, I would learn to cook." -> {"corrected": "If I had more time, I would learn to cook.", "corrections": [], "reply": "That sounds like a useful skill. I learned to cook last year, and it saves me a lot of money. What dish would you like to cook first?"}
Student: "The project was completed by our team last month." -> {"corrected": "The project was completed by our team last month.", "corrections": [], "reply": "Congratulations! What was your role in it?"}"""}

PROMPT_MODE = "full"     # "full" — каталог правил в промпте (базовая модель); "short" — без каталога (дообученная модель)

def use_fewshot():
    """Примеры в промпте — только для базовой модели (full). Дообученная модель учится на данных, примеры ей не нужны."""
    return USE_FEWSHOT and PROMPT_MODE == "full"

def rules_text(level):
    if PROMPT_MODE == "short":
        return f"- any rule_id from the LinguaBot grammar catalog for level {level}\n- other: none of the catalog rules"
    rs = [r for r in RULES if level in r["levels"] and r["wrong"]]
    return "\n".join(f'- {r["rule_id"]}: {r["name"]}. Example: "{r["wrong"]}" -> "{r["right"]}"' for r in rs) + "\n- other: none of the rules above"

OUTPUT = """Reply ONLY with JSON, no extra text:
{{"corrected": "<the whole student reply with ALL corrections applied>", "corrections": [{{"original": "<wrong fragment>", "correct": "<corrected fragment>", "rule_id": "<one id from the list below>"}}], "reply": "<your message to the student>"}}
Rules:
1. "corrected" must be the full student reply with every correction applied. If there are no mistakes, it must be an exact copy of the student reply.
2. "corrections" must list ALL language mistakes (grammar, vocabulary, spelling), in the order they appear in the student reply. There is no limit on the number of corrections. If there are no mistakes, use an empty list.
3. Each correction has exactly ONE rule_id. Use the most specific rule_id for the mistake.
4. Put only the wrong fragment in "original", not the whole sentence.
5. Never correct facts, opinions, or what the student chose to say. Do not correct short or one-word answers that are acceptable in conversation.
6. Do not mention the correction inside "reply".
7. Do not treat capitalization, missing final punctuation,
   missing apostrophes, or common informal chat expressions
   (idk, tbh, nah, gonna, wanna) as errors by themselves.
   Do not add cosmetic-only changes to "corrections".
   Preserve the student's meaning and informal tone.
   Correct genuine grammar, vocabulary, and spelling mistakes.

Allowed rule_id values:
{rules}{fewshot}"""

SYS_SCENARIO = """You play a role in an English practice dialogue with a Russian-speaking student. Your role: {bot_role}. Student level: {level}.
The student's goal: {goal}.
Conversation topics in order: {topics}. Current topic: {current_topic}.
Target vocabulary (use these words and encourage the student to use them): {vocabulary}.
Target grammar (encourage the student to use it): {grammar}.
{style}
Keep the conversation natural. Usually ask one relevant follow-up question. You may ask two closely related questions when useful, but never more than two. A question is not required in every reply. Prefer open questions (what, where, why, how) that encourage the student to use the target grammar. Do not repeat questions already asked. If the student goes off-topic, answer briefly and return to the current topic. If the student writes in Russian, answer in English.
Do not use emojis or markdown, do not write labels like "Follow-up question".
Error correction: {policy}
""" + OUTPUT

SYS_FREE = """You are a friendly English tutor chatting freely with a Russian-speaking student. Student level: {level}.
The student chooses the topic and may change it at any time.
{style}
Keep the conversation natural. Usually ask one relevant follow-up question. You may ask two closely related questions when useful, but never more than two. A question is not required in every reply. Prefer open questions (what, where, why, how). Do not repeat questions already asked. If the student asks you to suggest a topic, offer 3 topics suitable for the level.
If the topic is inappropriate (violence, weapons, explicit content, hate), politely refuse in one sentence and offer another topic.
If the student writes in Russian, answer in English. Do not use emojis or markdown, do not write labels like "Follow-up question".
Error correction: {policy}
""" + OUTPUT

HISTORY_JSON = True     # реплики бота в истории передаём в том же JSON-формате, что и ответ
                        # (иначе через 2-3 хода модель копирует свои прошлые реплики и перестаёт отвечать JSON)

def as_json_turn(text):
    obj = parse_json(text)
    if obj is not None and "reply" in obj:
        if "corrected" not in obj:
            obj["corrected"] = ""
        return json.dumps(obj, ensure_ascii=False)
    return json.dumps({"corrected": "", "corrections": [], "reply": text}, ensure_ascii=False)

def format_history(history):
    if not HISTORY_JSON:
        return list(history)
    return [{"role": m["role"], "content": as_json_turn(m["content"]) if m["role"] == "assistant" else m["content"]}
            for m in history]

QUOTES = str.maketrans({"`": "'", "´": "'", "‘": "'", "’": "'", "ʼ": "'"})

def normalize_text(text):
    """Апострофы с русской раскладки (i`m, don’t) -> обычный ' . Backend делает то же самое."""
    return str(text).translate(QUOTES).strip()

def mode_of(c):
    """Режим определяется по scenario_id: есть — сценарий, нет — свободный диалог. Поле mode необязательно."""
    mode = "scenario" if c.get("scenario_id") else "free"
    if c.get("mode") not in (None, mode):
        raise ValueError(f'{c.get("id", "")}: mode="{c["mode"]}", а scenario_id={c.get("scenario_id")!r}')
    return mode

def build_request(c):
    """(system, history, user) — ровно то, что получает модель."""
    b = band(c["level"])
    if mode_of(c) == "scenario":
        s = SCEN[c["scenario_id"]]
        system = SYS_SCENARIO.format(
            bot_role=s["bot_role"], level=c["level"], goal=s["goal"], topics="; ".join(s["topics"]),
            current_topic=s["topics"][c.get("topic_index", 0)], vocabulary=", ".join(s["target_vocabulary"]),
            grammar="; ".join(RULE_BY_ID[r]["name"] for r in s["target_rules"]),
            style=STYLE[b], policy=POLICY[b], rules=rules_text(c["level"]), fewshot=FEWSHOT[b] if use_fewshot() else "")
        history = c.get("history", [])
        if not history or history[0]["role"] != "assistant":   # история с реплики бота = середина диалога, opening_line не нужен
            history = [{"role": "assistant", "content": s["opening_line"]}] + history
    else:
        system = SYS_FREE.format(level=c["level"], style=STYLE_FREE[b], policy=POLICY[b], rules=rules_text(c["level"]), fewshot=FEWSHOT[b] if use_fewshot() else "")
        history = c.get("history", [])
    history = [{"role": m["role"], "content": normalize_text(m["content"]) if m["role"] == "user" else m["content"]}
               for m in history]
    return system, format_history(history), normalize_text(c["input"])

def training_example(c, target):
    """Один обучающий пример для SFT: те же system/history/user, что получает модель, + эталонный ответ."""
    system, history, user = build_request(c)
    answer = json.dumps(target, ensure_ascii=False)
    # prompt = вопрос (всё, что модель видит), completion = правильный ответ.
    # train_qlora.py учит модель только на completion; реплики бота в истории — контекст, на них не учимся.
    return {"prompt": [{"role": "system", "content": system}] + list(history) + [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": answer}]}

#  ВЫЗОВ МОДЕЛИ 
def ask(system, user, history, max_new_tokens=500):
    messages = [{"role": "system", "content": system}] + list(history) + [{"role": "user", "content": user}]
    text = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tok([text], return_tensors="pt").to(_model.device)
    t0 = time.time()
    out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][inputs.input_ids.shape[1]:]
    return _tok.decode(new, skip_special_tokens=True), round(time.time() - t0, 2), len(new)

def parse_json(answer, lenient=False):
    """lenient=True: терпимо к обёртке ```json и тексту вокруг (для приложения); False — строго (для метрик)."""
    text = answer.strip()
    if lenient:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.S)
        text = m.group(0) if m else text
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

def correction_card(corrections, level):
    """Карточка исправления для ученика: только для объясняемых правил, доступных на уровне.
    Порядок и лимит (1 для A1-A2, 2 для B1-B2) применяет backend до вызова этой функции."""
    ru = level in ("A1", "A2")
    lines = []
    for c in corrections:
        rule_id = c.get("rule_id")
        if rule_id in NON_EXPLAINABLE:
            continue
        rule = RULE_BY_ID.get(rule_id, {})
        if not rule or level not in rule.get("levels", []):
            continue
        text = rule.get("text_ru") if ru else rule.get("text_en")
        head = ("Ошибка: {} → Правильно: {}" if ru else "Mistake: {} -> Correct: {}").format(c.get("original"), c.get("correct"))
        lines.append(head + (f"\n{text}" if text else ""))
    return "\n".join(lines)
