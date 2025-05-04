import subprocess
import sys
def serve_model(model_uri: str, port: int = 5001):
    """
    Serve the MLflow model on the specified port using the mlflow models serve command.
    
    Args:
        model_uri (str): The URI of the model (e.g., runs:/<run_id>/model).
        port (int): The port to serve the model on (default: 5001).
    """
    try:
        # Construct the MLflow serve command
        cmd = [
            "mlflow", "models", "serve",
            "-m", model_uri,
            "-p", str(port),
            "--no-conda"
        ]
        print(f"Serving model at {model_uri} on port {port}...")
        # Run the command and keep the server running
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()  # Wait for the process to complete (server will run until interrupted)
    except FileNotFoundError:
        print("Error: MLflow is not installed or not found in the system PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error serving model: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopping model server...")
        process.terminate()
        process.wait()



# testing
# ml_run_path = get_tracking_uri()
# best_run_id = "f062432e10b54b24a5fbac1ba72893f0"
# mlflow.set_tracking_uri(get_tracking_uri())
# run = mlflow.get_run(best_run_id)
# experiment_id = run.info.experiment_id
# best_model_path = f"{ml_run_path}/{experiment_id}/{best_run_id}/artifacts/model"
# serve_model(best_model_path, port=5001)