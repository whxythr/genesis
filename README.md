# audit-env — RL-среда для обучения ИИ-аудитора

Детерминированный симулятор бухгалтерии **вымышленной** компании за один отчётный период.
Генератор строит согласованную скрытую историю (книга сходится), вносит контролируемые ошибки
с записанной причиной, агент за ≤100 шагов исследует записи через 5 инструментов и отправляет
`findings`, а оценщик **в отдельном процессе** сравнивает их со скрытым эталоном и выдаёт
терминальную награду.

Это среда, а не модель: ни обучения весов, ни LLM, ни сети внутри. Только стандартная библиотека
Python 3.11 (pytest нужен лишь для тестов).

- 4 типа ошибок: `duplicate_invoice`, `unapplied_payment`, `fx_error`, `period_cutoff`
- 2 сценария взаимодействующих ошибок, чистые компании и корректные «подозрительные» случаи
- независимый валидатор данных и проверка достаточности доказательств на каждой генерации
- открытый режим (`train`) и закрытый (`eval`), изоляция эталона, байт-в-байт воспроизводимость,
  инвариантность к порядку строк и смене ID

Подробное устройство, все решения с причинами, разбор сбоев с цифрами и ограничения — в
[docs/DESIGN.md](docs/DESIGN.md).

## Быстрый старт

### Docker (как будут проверять)

```bash
docker build -t audit-env .
docker run --rm --network none --cpus 2 --memory 4g audit-env make check
```

Сборке нужна сеть (базовый образ и pytest), запуску — нет.

### Локально

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
make check
```

Windows без `make`: `.\scripts\check.ps1` (то же самое, что `make check`).

`make check` завершается ненулевым кодом при любой ошибке и делает:

1. `pytest -q` — 261 тест: контракт, изоляция, генератор, matcher, атаки на награду,
   детерминизм, инвариантность, сценарии, а также **дословные проверки жюри** через
   `judge_adapter.py`: C01–C20, тесты награды 1–20 и примеры M01–M12, матрица D01–D20,
   парные проверки P01–P20 (`tests/test_judge_*.py`);
2. валидацию 10 train + 10 dev эпизодов: инварианты, совпадение с эталоном, достаточность
   доказательств, **байт-в-байт** совпадение с повторной генерацией;
3. прогон 4 базовых политик (train — открытый режим, dev — закрытый eval) с записью траекторий;
4. таблицу диагностики P / R / F1 / награда / шаги / время по политикам;
5. 60 парных проверок: эпизод против его копии с перемешанными строками и новыми ID.

## Подключение жюри: `judge_adapter.py`

Рабочая директория — корень репозитория (в Docker `/app`). Адаптер импортируется как
`import judge_adapter` (он сам добавляет `src/` в `sys.path`), сеть и доп. зависимости не нужны.

```python
import judge_adapter as ja

ja.generate_episode(seed, {"team_config": {...}}, "/jury/ep1")   # открытые файлы + truth.json + provenance.json
env = ja.make_env()
obs, info = env.reset(None, {"mode": "closed", "episode_dir": "/jury/ep1", "team_config": {}})
obs, reward, terminated, truncated, info = env.step({"name": "list_records",
                                                     "args": {"table": "ledger", "cursor": None, "limit": 50}})
ja.score_episode("/jury/ep1", findings, inspection_steps)  # {"tp","fp","fn","precision","recall","f1","return"}
```

| Что | Как |
| --- | --- |
| `generate_episode(seed, config, output_dir)` | `config` — `{"team_config": {...}}` или сами ключи конфига; пишет `manifest.json policy.json ledger.csv invoices.jsonl payments.jsonl fx.csv truth.json provenance.json` (закрытые файлы можно также положить в `<dir>/closed/`) |
| `reset(seed, config)` | `mode: "closed"` (= наш `eval`) или `"train"`; с `episode_dir` грузит **именно этот** эпизод (повторный reset — заново), без него в train генерирует по `seed` + `team_config` |
| первое наблюдение | `{"task", "manifest", "policy", "remaining_steps": 100}`; описание инструментов — в тексте `task` |
| следующие наблюдения | `{"result": value \| null, "error": null \| {"code", "message"}, "remaining_steps"}` |
| `list_records` | args `table` ∈ `ledger`, `invoices`, `payments`, `fx`; `cursor` (первый — `null`); `limit` 1…50; опц. `filter`. Результат `{"records": [...], "next_cursor": str \| null}`; у счетов в списке — сводка без `line_items` |
| `get_record` | args `table`, `id`: ledger — `row_id`; invoices — `invoice_id` (или `doc_id`); payments — `payment_id` (или `doc_id`); fx — `fx_id` (или `date:source:target`). Результат — объект записи |
| `get_document` | args `doc_id` счёта или платежа. Результат — полный документ |
| `calculator` | args `expression` (`+ - * / ( )`, десятичные). Результат — десятичная строка, `0.1+0.2` → `"0.3"` |
| `submit` | args `{"findings": <findings.json>}`. Результат `{"accepted": true}` или ошибка `INVALID_FINDINGS` (структура), без оценки правильности |
| награда / info | train: терминальная награда и метрики в `info`; closed: `reward = 0.0` всегда, `info = {}` |
| `score_episode` | закрытый вызов на стороне жюри; невалидная структура → `return = -1`, остальные поля `null`, запись в `runs/judge_grader_log.jsonl` (путь меняется переменной `AUDIT_ENV_GRADER_LOG`) |

Четыре типа ошибок (точные строки `type`): `duplicate_invoice`, `unapplied_payment`, `fx_error`,
`period_cutoff`. Каноничные `affected_ids` и суммы — в [docs/DESIGN.md](docs/DESIGN.md) (D9, D12, D13).
Настройки генератора для каждого закрытого сценария D01–D20 (чистые с разбитыми платежами,
возвратами, подписками, валютной/периодной границей; каждый тип по отдельности, включая
пропущенное vs частичное распределение; оба сценария взаимодействия) —
[configs/judge_matrix.json](configs/judge_matrix.json).

## Использование из Python

```python
from audit_env.env.core import AuditEnv

env = AuditEnv(mode="train")                     # "eval": reward=0 и пустой info до finalize()
obs, info = env.reset(seed=123, config={})       # или config={"episode": {"split": "train", "episode_id": "EP-..."}}

obs, reward, terminated, truncated, info = env.step(
    {"tool": "list_records", "args": {"table": "payments", "limit": 50}})

obs, reward, terminated, truncated, info = env.step({"tool": "submit", "args": {"findings": {
    "episode_id": env.episode_id,
    "findings": []}}})
env.close()
```

`reset(seed, config)`: `config` — переопределения `configs/default.json` (генерация «на лету»
в процессе оценщика) или ссылка на готовый эпизод на диске. Контракт JSON-совместимый, в стиле
Gymnasium.

**Первое наблюдение:** `task` (текст задачи и определения ошибок), `manifest`, `policy`
(правила периода, курсов, округления, проводок, дублей), `tools`, `steps_remaining`.

**Действия** — ровно пять:

| tool | args | что возвращает |
| --- | --- | --- |
| `list_records` | `table` ∈ ledger/invoices/payments/fx, `cursor`, `limit` ≤ 50, опц. `filter` | страница записей, `total`, непрозрачный стабильный `next_cursor` |
| `get_record` | `table`, `id` | одна запись |
| `get_document` | `doc_id` | полный документ (у счетов — с `line_items`) |
| `calculator` | `expression` ≤ 200 символов, `+ - * / ( )` | десятичная строка (свой парсер, без `eval`) |
| `submit` | `findings` | завершает эпизод |

Любая ошибка ввода — фиксированный код (`UNKNOWN_ACTION`, `BAD_ARGS`, `UNKNOWN_TABLE`, `NOT_FOUND`,
`BAD_CURSOR`, `CALC_ERROR`, …) без эха ввода, и она **расходует шаг**. 100 шагов включая submit;
лимит шагов или 60 с → `truncated=True` и оценка пустого ответа.

**Ответ** (`schemas/findings.schema.json`, ≤30 выводов, ≤10 доказательств, explanation ≤1000):

```json
{"episode_id": "EP-...", "findings": [
  {"type": "fx_error", "affected_ids": ["PAY-..."], "claimed_amount_minor": 108,
   "currency": "USD", "evidence_ids": ["D-...", "FX-...", "T-..."], "explanation": "..."}]}
```

**Награда** (только терминальная, по ходу эпизода 0):

```
TP — совпали тип, множество affected_ids, валюта (всё — точные строки, регистр и пробелы значимы),
     сумма точно (допуск τ = 0 для всех типов), и evidence_ids содержит полный допустимый набор
     доказательств, причём все evidence_ids существуют в открытых записях
сопоставление — максимальное паросочетание; повторный вывод — отдельный FP
F1 = 2TP / (2TP + FP + FN);  пустой эталон и пустой ответ → F1 = 1
reward = clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1);  неверная структура → −1
inspection_steps = шаги до submit (включая ошибочные), сам submit не считается
```

Неверная структура (= −1): больше 30 выводов, больше 10 `evidence_ids` в выводе (считается до
удаления повторов), `explanation` длиннее 1000 символов Unicode, отсутствующее поле, неверный JSON-тип
(например `claimed_amount_minor` = `"10000"`, `true` или `1.5`), лишний ключ, чужой `episode_id`.
Структурно верный, но неправильный вывод (неизвестный тип, чужие ID, валюта, сумма) — просто FP.
Код matcher'а с нормализацией, допусками и выбором при неоднозначности:
[src/audit_env/grader/](src/audit_env/grader/) (`normalize.py`, `matcher.py`, `metrics.py`, `reward.py`).

## Данные

| Путь | Что |
| --- | --- |
| `episodes/{train,dev}/EP-*/` | открытые файлы: `manifest.json`, `policy.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv` |
| `grader_data/{train,dev}/EP-*/` | закрытые: `truth.json`, `provenance.json` (seed, конфиг, план, каждое изменение «было → стало», дифф книги) |
| `configs/default.json`, `configs/{train,dev}/v1.json` | параметры генератора; каждый эпизод несёт `config_id` = имя + хеш конфига |
| `trajectories/train/`, `trajectories/dev_eval/` | `trajectories.jsonl` (строка на шаг), `summary.json`, журнал оценщика |
| `schemas/` | JSON Schema: manifest, action, observation, findings, truth, trajectory |

Деньги — целые minor units (JPY 0, USD/EUR/KZT/GBP 2, KWD 3 знака), курсы — десятичные строки,
расчёт через `Decimal` с `ROUND_HALF_UP`. Время — ISO-8601 со смещением; период `[start, end)`
в поясе компании. Отсутствующая связь — явный `null` (в CSV — пустое поле).

## Команды

| Команда | Что делает |
| --- | --- |
| `make check` | всё вышеперечисленное |
| `make demo` | скриптовый аудитор проходит эпизод шаг за шагом, затем P/R/F1/награда и вердикт по каждому выводу |
| `make generate` / `make validate` | пересоздать / проверить train и dev |
| `make baselines` | 4 политики: train открыто, dev закрыто |
| `make invariance` | парные проверки порядка строк и ID |
| `python -m audit_env transform --split dev --seed 7` | копия набора с перемешанными строками и новыми ID (маппинг — только у оценщика) |
| `python scripts/failure_analysis.py 100 [--naive \| --ledger-amounts]` | разбор TP/FP/FN по типам ошибок |

## Базовые политики (из `make check`)

| Политика | train (открыто) | dev (eval) |
| --- | --- | --- |
| `oracle` — эталон, получает truth от harness | 1.000 | 1.000 |
| `reconcile` — правила сверки, только через инструменты | 0.957 | 0.957 |
| `random_explorer` — случайные шаги, seed фиксирован | −0.607 | −0.605 |
| `flag_all` — атака «пометить всё» | −1.000 | −1.000 |

## Структура

```
src/audit_env/
  money.py timeutil.py ids.py schema.py config.py jsonutil.py
  generator/  company.py posting.py scenarios.py errors.py invariants.py evidence.py writer.py episode.py
  env/        core.py tools.py store.py calculator.py
  grader/     service.py client.py core.py normalize.py matcher.py metrics.py reward.py
  policies/   oracle.py reconcile.py random_explorer.py flag_all.py
  runner.py transform.py demo.py cli.py
judge_adapter.py   # точка подключения жюри
tests/ scripts/ schemas/ configs/ episodes/ grader_data/ trajectories/ docs/
```

## Лицензия

MIT, см. [LICENSE](LICENSE). Все компании, люди и суммы вымышлены.
