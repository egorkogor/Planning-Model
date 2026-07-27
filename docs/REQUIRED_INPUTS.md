# Required inputs

До P00 нужны:

- этот git repository;
- Python 3.11+;
- Builder Agent с shell/filesystem/git;
- локальная машина либо API/credentials разрешённого cloud provider;
- возможность создать отдельные Data Sealer, Evaluation Runner и Audit Agent environments;
- доступ к Hugging Face в P02 для immutable model locks.

Технические параметры, split sizes, model architecture, training grid, prompt selection и gates уже находятся в `docs/`. Оператор не должен выбирать их вручную.

Confirmatory run нельзя проводить на одном agent/environment без Data Sealer/Evaluator separation: отсутствие evaluator separation делает его invalid.
