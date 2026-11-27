"""Register the flows as scheduled Prefect deployments and serve them.

Run `prefect server start` in another terminal first, then:
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
    python deploiement_prefect.py
"""

from prefect import serve

from pipeline_prefect import (
    flow_all,
    flow_evaluate,
    flow_predict,
    flow_prepare,
    flow_test,
    flow_train,
)

if __name__ == "__main__":
    serve(
        # Full pipeline every day at 02:00
        flow_all.to_deployment(name="ml-pipeline-all", cron="0 2 * * *"),
        # Retrain every Monday at 03:00
        flow_train.to_deployment(name="ml-pipeline-train", cron="0 3 * * 1"),
        # On demand / hourly checks
        flow_test.to_deployment(name="ml-pipeline-test", interval=3600),
        flow_prepare.to_deployment(name="ml-pipeline-prepare"),
        flow_evaluate.to_deployment(name="ml-pipeline-evaluate"),
        flow_predict.to_deployment(name="ml-pipeline-predict"),
    )
