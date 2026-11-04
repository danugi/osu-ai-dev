# osu-ai-dev — переработка Этапа 2 (стадии 2.1–2.5)

База: коммит 9aecb61 (17.10). Итог: 27 файлов, +603/−708 строк. Наблюдаемое
поведение среды/судейства/оракула НЕ изменено — проверено байт-в-байт
смоуками (рендер симулятора и траектория Учителя v3 на фиксированной карте
и сидах совпадают с эталоном до переработки).

## Что изменилось

| Стадия | Что |
|---|---|
| 2.1 | Пыль: мёртвый код (screen_to_osu, ветка 30Гц в env, тернары, note), оживлена потерянная метрика value-loss в dagger, врущие комментарии исправлены, pf_calib.json убран из git (.gitignore) |
| 2.6a | 30Hz-патчи diagnose/servo удалены (30Гц-модели мертвы) |
| 2.2 | **core.py** — единый источник констант (TOSU_URL, скорости, награды, DT_MS); **models.py** — реестр моделей + infer_stack + инвентаризация (бывший census.py); **cli.py** — REPL-хелперы. Разорван импорт sim_env → env: **тренировка больше не требует Windows** |
| 2.3 | **oracle.py** (Учитель v3 + aux_target) и **net.py** (BCNet + перенос весов в обе стороны) — вынесены из bc_train дословно; удалены reverse_check.py и logit_probe.py; Δlogits-гейт встроен в самопроверку bc_train |
| 2.4 | **viewer.py** — бывший standalone-просмотрщик vision.py; из ядра зрения убраны user32/requests/time/os/copy/bisect с импорт-пути; AR-формула в парсере — через ar_ms_from_value |
| 2.5 | Дедуп: slider_pos (3 копии → 1 в vision), decode_move (4 копии → 1 в core), circle_radius_osu и od_from_w50 (по 3 копии → 1 в vision), cur_od заменён пробросом od |

## Найден и исправлен баг апстрима

`dagger_train.py` (был уже в 9aecb61), обратный перенос весов PPO → BCNet:

    net.btn_head.bias.copy_(pol.action_net.bias[27])   # скаляр вместо среза!

`copy_` бродкастил ОДИН элемент на все 5 bias'ов кнопок. Argmax-гейты этого
не видят (равномерный сдвиг 5 логитов не меняет softmax), поэтому баг жил
тихо и портил bias кнопок при каждой итерации DAgger. Одно-off пробник
reverse_check.py имел правильный срез — ошибка была только в продакшн-коде.
Исправлено в net.py: `ppo_to_bcnet`.

## Что сделать руками после распаковки

1. Удалить из папки три устаревших файла (в архив их не включили, но при
   распаковке поверх старой папки они могли остаться):
   `reverse_check.py`, `logit_probe.py`, `census.py`.
2. Если play_udp.py / live_oracle.py / dagger_train_offline.py (файлы
   Task 21) ругаются на `from bc_train import BCNet, oracle_policy` —
   теперь так: `from net import BCNet` и `from oracle import oracle_policy`.
   (env.py, vision.py, sim_env.py, input_ctrl.py не меняли контракт.)
3. Авто-стек: diagnose_v2 и servo_v2 больше не спрашивают стек — читают из
   модели (аргумент n_stack остался для совместимости, 0 = авто).

## Внимание: версия stable-baselines3

В песочнице на sb3 2.9.2 обнаружено: обёртки VecFrameStack+VecTransposeImage
дают float observation space, и PPO перестаёт сам делить кадры на 255 —
семантика обучения меняется по сравнению со старыми версиями sb3.
Зафиксируй свою рабочую версию, под которой учились текущие модели:

    pip freeze | findstr stable-baselines3

и ставь именно её на новых машинах (или пришли вывод — добавим requirements.txt).

## Как проверялось (в Linux-песочнице, без Windows)

- py_compile 22/22 файлов;
- импорт core/models/cli/oracle/net/vision/sim_env/bc_train/dagger_train
  без единой Windows-заглушки (раньше обучение тянуло ctypes.windll);
- рендер-смоук: 675 кадров SimOsuEnv на фиксированной карте — байт-в-байт
  с эталоном ДО переработки (награда совпадает до 6 знака);
- оракул-смоук: траектория Учителя v3 (891 кадр, reward 23.478773) —
  байт-в-байт с эталоном;
- перенос весов: BCNet → PPO → BCNet — все обучаемые веса равны с точностью
  до бита (кроме aux_head, которого в PPO нет by design);
- каждая стадия — отдельный коммит git (ветка stage2), откатываемо.

Рантайм-части, требующие Windows (захват tosu, SendInput, окно Neuro Vision),
проверяются на твоей машине штатным запуском: bc_train → train_stage → play.
