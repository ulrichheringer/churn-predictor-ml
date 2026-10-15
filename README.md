# churn-predictor-ml

Prevê churn SaaS com dados sintéticos de uso, suporte, tempo de conta e plano, usando regressão logística.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout estratificado de 450 contas — acurácia **0,6711**, precisão **0,2515**, recall **0,6462**, F1 **0,3621**, ROC AUC **0,7516**. O peso balanceado favorece recall e produz falsos positivos; a regra sintética de churn não representa cancelamentos reais.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
