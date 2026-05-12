from prefect import task
import subprocess

@task
def run_dbt_models():
    result = subprocess.run(
        ["dbt", "run", "--project-dir", "/app/dbt", "--profiles-dir", "/app/dbt"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"dbt run failed: {result.stderr}")
    print(result.stdout)