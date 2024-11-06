import argparse
import os
import shutil
from zipfile import ZipFile

import pandas as pd
from matplotlib import pyplot as plt


def connect_to_h2o_engine(token: str, client_id, token_endpoint_url, environment):
    # https://internal.dedicated.h2o.ai/cli-and-api-access
    """Establishes a secure connection to the H2O Engine Manager using the provided token."""
    import h2o_authn
    token_provider = h2o_authn.TokenProvider(
        refresh_token=token,
        client_id=client_id,
        token_endpoint_url=token_endpoint_url,
    )

    import h2o_engine_manager
    engine_manager = h2o_engine_manager.login(
        environment=environment,
        token_provider=token_provider
    )

    # https://docs.h2o.ai/mlops/py-client/install
    # os.system('pip install h2o-mlops')
    # import h2o_mlops
    # mlops = h2o_mlops.Client(
    #  gateway_url="https://mlops-api.internal.dedicated.h2o.ai",
    #    token_provider=token_provider
    # )

    print("Successfully connected to H2O engine manager.")
    return engine_manager


def connect_to_driverless_ai(engine_manager, dai_engine: str = None):
    """Creates a Driverless AI engine and establishes a connection to it."""
    dai_engine_obj = None
    for dai_inst in engine_manager.dai_engine_client.list_all_engines():
        if dai_inst.display_name == dai_engine:
            dai_engine_obj = engine_manager.dai_engine_client.get_engine(dai_engine)
            if dai_engine_obj.state.value != "STATE_RUNNING":
                print(f"Waking up instance {dai_engine}")
                dai_engine_obj.resume()
                dai_engine_obj.wait()

    if dai_engine_obj is None:
        # if DAI Engine does not exist
        print(f"Creating instance {dai_engine}")
        dai_engine_obj = engine_manager.dai_engine_client.create_engine(display_name=dai_engine)
        dai_engine_obj.wait()

    dai = dai_engine_obj.connect()
    print(f"Successfully connected to Driverless AI engine: {dai_engine}")
    return dai


def create_dataset(dai, data_url: str, dataset_name: str, data_source: str = "s3", force: bool = True):
    """Creates a dataset in the Driverless AI instance."""
    dataset = dai.datasets.create(
        data=data_url,
        data_source=data_source,
        name=dataset_name,
        force=force
    )
    print(f"Dataset {dataset_name} with reusable dataset_key: {dataset.key} created successfully.")
    return dataset


def split_dataset(dataset, train_size: float, train_name: str, test_name: str,
                  target_column: str, seed: int = 42):
    """Splits a dataset into train and test sets."""
    dataset_split = dataset.split_to_train_test(
        train_size=train_size,
        train_name=train_name,
        test_name=test_name,
        target_column=target_column,
        seed=seed
    )

    print("Dataset successfully split into training and testing sets.")
    for k, v in dataset_split.items():
        print(f"Name: {v.name} with reusable dataset_key: {v.key}")

    return dataset_split


def create_experiment(dai, dataset_split, target_column: str, scorer: str = 'F1',
                      task: str = 'classification', experiment_name: str = 'Experiment',
                      accuracy: int = 1, time: int = 1, interpretability: int = 6,
                      fast=True,
                      force: bool = True):
    """Creates an experiment in Driverless AI."""
    experiment_settings = {
        **dataset_split,
        'task': task,
        'target_column': target_column,
        'scorer': scorer
    }

    dai_settings = {
        'accuracy': accuracy,
        'time': time,
        'interpretability': interpretability,
    }
    if fast:
        print("Using fast settings, but still making autoreport")
        dai_settings.update({
            'make_python_scoring_pipeline': 'off',
            'make_mojo_scoring_pipeline': 'off',
            'benchmark_mojo_latency': 'off',
            'make_autoreport': True,
            'check_leakage': 'off',
            'check_distribution_shift': 'off'
        })

    experiment = dai.experiments.create(
        **experiment_settings,
        name=experiment_name,
        **dai_settings,
        force=force
    )

    print(f"Experiment {experiment_name} with reusable experiment_key: {experiment.key} created with settings: "
          f"Accuracy={accuracy}, Time={time}, Interpretability={interpretability}")
    return experiment


def get_experiment_from_key(experiment_key, token, client_id, token_endpoint_url, dai_engine, environment):
    # FIXME: not used yet, would be used to act more on experiment, like restart etc.
    # Connect to the engine manager and Driverless AI
    engine_manager = connect_to_h2o_engine(token, client_id, token_endpoint_url, environment)
    dai = connect_to_driverless_ai(engine_manager, dai_engine)

    # Get the experiment
    experiment = dai.experiments.get(experiment_key)
    return experiment


def visualize_importance(experiment):
    """Visualizes and saves variable importance plot."""
    var_imp = experiment.variable_importance()
    print("\nVariable Importance Output:")
    print(var_imp)

    # Save variable importance to csv
    df = pd.DataFrame(var_imp.data, columns=var_imp.headers)
    csv_file = "variable_importance.csv"
    df.to_csv(csv_file, index=False)
    df_top10 = df.sort_values('gain', ascending=False).head(10)

    plt.figure(figsize=(12, 8))
    plt.barh(df_top10['description'], df_top10['gain'])
    plt.title('Top 10 Important Variables')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()

    output_path = 'variable_importance.png'
    plt.savefig(output_path)
    print(f"\nVariable importance plot saved as {output_path} and csv file as {csv_file}")

    print("\nTop 10 Important Variables:")
    print(df_top10[['description', 'gain']].to_string(index=False))


def print_experiment_details(experiment):
    """Prints details of a Driverless AI experiment."""
    print(f"\nExperiment Details:")
    print(f"Name: {experiment.name}")
    print("\nDatasets:")
    for dataset in experiment.datasets:
        print(f" - {dataset}")
    print(f"\nTarget: {experiment.settings.get('target_column')}")
    print(f"Scorer: {experiment.metrics().get('scorer')}")
    print(f"Task: {experiment.settings.get('task')}")
    print(f"Size: {experiment.size}")
    print(f"Summary: {experiment.summary}")
    print("\nStatus:")
    print(experiment.status(verbose=2))
    print("\nWeb Page: ", end='')
    experiment.gui()

    print(f"\nMetrics: {experiment.metrics()}")


def plot_roc_curve(roc_data, save_dir='plots'):
    """Plot ROC (Receiver Operating Characteristic) curve and save to file"""
    df = pd.DataFrame(roc_data['layer'][0]['data']['values'])

    plt.figure(figsize=(8, 6))
    plt.plot(df['False Positive Rate'], df['True Positive Rate'], 'b-', label='ROC curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall(pr_data, save_dir='plots'):
    """Plot Precision-Recall curve and save to file"""
    df = pd.DataFrame(pr_data['layer'][0]['data']['values'])

    plt.figure(figsize=(8, 6))
    plt.plot(df['Recall'], df['Precision'], 'g-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_gains_chart(gains_data, save_dir='plots'):
    """Plot Cumulative Gains chart and save to file"""
    df = pd.DataFrame(gains_data['layer'][0]['data']['values'])

    plt.figure(figsize=(8, 6))
    plt.plot(df['Quantile'], df['Gains'], 'b-')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('Population Percentage')
    plt.ylabel('Cumulative Gains')
    plt.title('Cumulative Gains Chart')
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'gains_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_lift_chart(lift_data, save_dir='plots'):
    """Plot Lift chart and save to file"""
    df = pd.DataFrame(lift_data['layer'][0]['data']['values'])

    plt.figure(figsize=(8, 6))
    plt.plot(df['Quantile'], df['Lift'], 'g-')
    plt.axhline(y=1, color='r', linestyle='--', label='Baseline')
    plt.xlabel('Population Percentage')
    plt.ylabel('Lift')
    plt.title('Lift Chart')
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'lift_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ks_chart(ks_data, save_dir='plots'):
    """Plot Kolmogorov-Smirnov chart and save to file"""
    df = pd.DataFrame(ks_data['layer'][0]['data']['values'])

    plt.figure(figsize=(8, 6))
    plt.plot(df['Quantile'], df['Gains'], 'b-')
    plt.xlabel('Population Percentage')
    plt.ylabel('KS Statistic')
    plt.title('Kolmogorov-Smirnov Chart')
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'ks_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_charts(roc_curve, prec_recall_curve, gains_chart, lift_chart, ks_chart, save_dir='plots'):
    """Plot all available classification metrics charts and save to file"""

    # Create subplots for available charts
    available_charts = sum(x is not None for x in [roc_curve, prec_recall_curve, gains_chart, lift_chart, ks_chart])
    rows = (available_charts + 1) // 2  # Calculate rows needed

    fig = plt.figure(figsize=(15, 5 * rows))

    plot_idx = 1

    if roc_curve is not None:
        plt.subplot(rows, 2, plot_idx)
        df = pd.DataFrame(roc_curve['layer'][0]['data']['values'])
        plt.plot(df['False Positive Rate'], df['True Positive Rate'], 'b-')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plot_idx += 1

    if prec_recall_curve is not None:
        plt.subplot(rows, 2, plot_idx)
        df = pd.DataFrame(prec_recall_curve['layer'][0]['data']['values'])
        plt.plot(df['Recall'], df['Precision'], 'g-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plot_idx += 1

    if gains_chart is not None:
        plt.subplot(rows, 2, plot_idx)
        df = pd.DataFrame(gains_chart['layer'][0]['data']['values'])
        plt.plot(df['Quantile'], df['Gains'], 'b-')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Population Percentage')
        plt.ylabel('Cumulative Gains')
        plt.title('Cumulative Gains Chart')
        plt.grid(True)
        plot_idx += 1

    if lift_chart is not None:
        plt.subplot(rows, 2, plot_idx)
        df = pd.DataFrame(lift_chart['layer'][0]['data']['values'])
        plt.plot(df['Quantile'], df['Lift'], 'g-')
        plt.axhline(y=1, color='r', linestyle='--')
        plt.xlabel('Population Percentage')
        plt.ylabel('Lift')
        plt.title('Lift Chart')
        plt.grid(True)
        plot_idx += 1

    if ks_chart is not None:
        plt.subplot(rows, 2, plot_idx)
        df = pd.DataFrame(ks_chart['layer'][0]['data']['values'])
        plt.plot(df['Quantile'], df['Gains'], 'b-')
        plt.xlabel('Population Percentage')
        plt.ylabel('KS Statistic')
        plt.title('Kolmogorov-Smirnov Chart')
        plt.grid(True)
        plot_idx += 1

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'all_classification_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def key_to_experiment(experiment_key, client_id, dai_engine, token_endpoint_url, token, environment):
    if experiment_key is None:
        raise ValueError("Either experiment or experiment_key must be provided")
    engine_manager = connect_to_h2o_engine(token, client_id, token_endpoint_url, environment)
    dai = connect_to_driverless_ai(engine_manager, dai_engine)
    experiment = dai.experiments.get(experiment_key)
    return experiment


def get_artifacts(experiment=None, experiment_key=None, client_id=None, dai_engine=None, token_endpoint_url=None,
                  token=None, environment=None, save_dir='./'):
    if experiment is None:
        experiment = key_to_experiment(experiment_key, client_id, dai_engine, token_endpoint_url, token, environment)

    artifacts = experiment.artifacts.list()
    if 'logs' in artifacts:
        logs_zip = experiment.artifacts.download(only=['logs'], dst_dir=save_dir, overwrite=True)['logs']
        logs_dir = './logs_dir'
        with ZipFile(logs_zip, 'r') as zip_ref:
            zip_ref.extractall(logs_dir)
        os.remove(logs_zip)
        log_files = [os.path.join(os.getcwd(), logs_dir, x) for x in os.listdir(logs_dir)]

        for fil in log_files:
            if fil.endswith('.zip'):
                with ZipFile(fil, 'r') as zip_ref:
                    zip_ref.extractall(logs_dir)
        log_files = [os.path.join(os.getcwd(), logs_dir, x) for x in os.listdir(logs_dir)]
        print(f"List of experiment log files extracted include: {log_files}")

        moved = []
        useful_extensions = ['.png', '.csv', '.json']
        for fil in log_files:
            if any(fil.endswith(ext) for ext in useful_extensions):
                shutil.copy(fil, save_dir)
                new_abs_path = os.path.join(save_dir, os.path.basename(fil))
                moved.append(new_abs_path)
        print(f"Log files moved to {save_dir} include: {moved}")

    if 'summary' in artifacts:
        summary_zip = experiment.artifacts.download(only=['summary'], dst_dir=save_dir, overwrite=True)['summary']
        summary_dir = './summary_dir'
        with ZipFile(summary_zip, 'r') as zip_ref:
            zip_ref.extractall(summary_dir)
        os.remove(summary_zip)
        summary_files = [os.path.join(os.getcwd(), summary_dir, x) for x in os.listdir(summary_dir)]
        print(f"List of summary log files extracted include: {summary_files}")
        moved = []
        useful_extensions = ['.png', '.csv', '.json']
        for fil in summary_files:
            if any(fil.endswith(ext) for ext in useful_extensions):
                shutil.copy(fil, save_dir)
                new_abs_path = os.path.join(save_dir, os.path.basename(fil))
                moved.append(new_abs_path)
        print(f"Summary files moved to {save_dir} include: {moved}")
    if 'train_predictions' in artifacts:
        train_preds = experiment.artifacts.download(only=['train_predictions'], dst_dir=save_dir, overwrite=True)[
            'train_predictions']
        print(f"Train predictions saved to {train_preds}")
        print(f"Head of train predictions: {pd.read_csv(train_preds).head()}")
    if 'test_predictions' in artifacts:
        test_preds = experiment.artifacts.download(only=['test_predictions'], dst_dir=save_dir, overwrite=True)[
            'test_predictions']
        print(f"Test predictions saved to {test_preds}")
        print(f"Head of test predictions: {pd.read_csv(test_preds).head()}")
    if 'autoreport' in artifacts:
        autoreport = experiment.artifacts.download(only=['autoreport'], dst_dir=save_dir, overwrite=True)['autoreport']
        print(f"Autoreport saved to {autoreport}")
    if 'autodoc' in artifacts:
        autodoc = experiment.artifacts.download(only=['autodoc'], dst_dir=save_dir, overwrite=True)['autodoc']
        print(f"Autoreport saved to {autodoc}")


def main():
    parser = argparse.ArgumentParser(description="Run Driverless AI experiments from command line.")

    # instance
    parser.add_argument("--engine", "--dai_engine", default=os.getenv('DAI_ENGINE', "daidemo"),
                        help="Name of the DAI engine")
    parser.add_argument("--client_id", "--dai_client_id", default=os.getenv('DAI_CLIENT_ID', "hac-platform-public"),
                        help="Name of client_id")
    parser.add_argument("--token_endpoint_url", "--dai_token_endpoint_url", default=os.getenv('DAI_TOKEN_ENDPOINT_URL',
                                                                                              "https://auth.internal.dedicated.h2o.ai/auth/realms/hac/protocol/openid-connect/token"),
                        help="Token endpoint url")
    parser.add_argument("--environment", "--dai_environment",
                        default=os.getenv('DAI_ENVIRONMENT', "https://internal.dedicated.h2o.ai"),
                        help="DAI environment")
    parser.add_argument("--token", "--dai_token", default=os.getenv('DAI_TOKEN'),
                        help="DAI token")
    parser.add_argument('--demo_mode', action='store_true', help="Use demo mode")

    # Existing experiment
    parser.add_argument("--experiment_key", default="",
                        help="Key of an existing experiment to re-use")
    parser.add_argument("--dataset_key", default="",
                        help="Key of an existing dataset to re-use")

    # Creating new dataset
    parser.add_argument("--data-url", required=False,
                        default="",
                        help="URL to the dataset (e.g., S3 URL)")
    parser.add_argument("--dataset-name", default="Dataset",
                        help="Name for the dataset in DAI (default: Dataset)")
    parser.add_argument("--data-source", default="s3",
                        help="Source type of the dataset (default: s3)")

    # Creating new experiment
    parser.add_argument("--target-column", "--target",
                        default="Churn?",
                        required=False,
                        help="Name of the target column for prediction")
    parser.add_argument("--task", default="classification",
                        choices=["classification", "regression", "predict",
                                 "shapley",
                                 "shapley_original_features",
                                 "shapley_transformed_features",
                                 "transform",
                                 "fit_transform",
                                 "fit_and_transform",
                                 "artifacts",
                                 ],
                        help="Type of ML task (default: classification)")
    parser.add_argument("--scorer", default="F1",
                        help="Evaluation metric to use (default: F1)")
    parser.add_argument("--experiment-name", default="Experiment",
                        help="Name for the experiment (default: Experiment)")
    parser.add_argument("--accuracy", type=int, choices=range(1, 11), default=1,
                        help="Accuracy setting (1-10, default: 1)")
    parser.add_argument("--time", type=int, choices=range(1, 11), default=1,
                        help="Time setting (1-10, default: 1)")
    parser.add_argument("--interpretability", type=int, choices=range(1, 11), default=6,
                        help="Interpretability setting (1-10, default: 6)")
    parser.add_argument("--train-size", type=float, default=0.8,
                        help="Proportion of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--fast", action="store_false",
                        help="Use fast settings for experiment or predictions")
    parser.add_argument("--force", action="store_false",
                        help="Force overwrite existing datasets/experiments")

    args = parser.parse_args()

    # Connect to H2O
    engine_manager = connect_to_h2o_engine(args.token, args.client_id, args.token_endpoint_url, args.environment)
    dai = connect_to_driverless_ai(engine_manager, args.engine)

    # Create plots directory if it doesn't exist
    save_dir = './'

    # Ensure all columns are displayed
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to multiple lines

    if args.experiment_key:
        # Re-use existing experiment
        experiment = dai.experiments.get(args.experiment_key)
        print(f"Re-using existing experiment: {experiment.name} with experiment_key: {experiment.key}")

        # Create dataset for (e.g.) transform or predict
        if args.data_url:
            dataset = create_dataset(
                dai,
                args.data_url,
                args.dataset_name,
                args.data_source,
                args.force
            )
        elif args.dataset_key:
            # Re-use existing dataset
            dataset = dai.datasets.get(args.dataset_key)
            print(f"Re-using existing dataset: {dataset.name} with dataset_key: {dataset.key}")
        else:
            dataset = None
        print(f"Performing task {args.task} on experiment {experiment.name}")
        if args.task == 'predict':
            if dataset is None:
                print("Dataset key is required for prediction.")
            else:
                prediction = experiment.predict(dataset)
                prediction_csv = prediction.download(dst_file=os.path.join(save_dir, 'prediction.csv'), overwrite=True)
                print(f"Prediction saved to {prediction_csv}")
                print(f"Head of prediction:\n{pd.read_csv(prediction_csv).head()}")
        elif args.task in ['shapley', 'shapley_original_features']:
            if dataset is None:
                print("Dataset key is required for shapley prediction.")
            else:
                prediction = experiment.predict(dataset, include_shap_values_for_original_features=True,
                                                use_fast_approx_for_shap_values=args.fast)
                prediction_csv = prediction.download(dst_file=os.path.join(save_dir, 'shapley_original_features.csv'),
                                                     overwrite=True)
                print(f"Shapley on original features saved to {prediction_csv}")
                print(f"Head of shapley on original features:\n{pd.read_csv(prediction_csv).head()}")
                print(
                    "Column names for contributions (Shapley values) are in form contrib_<original_column_name>, which you should programatically access instead of repeating all the names in any python code.")
        elif args.task == 'shapley_transformed_features':
            if dataset is None:
                print("Dataset key is required for shapley prediction.")
            else:
                prediction = experiment.predict(dataset, include_shap_values_for_transformed_features=True,
                                                use_fast_approx_for_shap_values=args.fast)
                prediction_csv = prediction.download(
                    dst_file=os.path.join(save_dir, 'shapley_transformed_features.csv'), overwrite=True)
                print(f"Shapley on transformed features saved to {prediction_csv}")
                print(f"Head of shapley on transformed features:\n{pd.read_csv(prediction_csv).head()}")
                print(
                    "Column names for contributions (Shapley values) are in form contrib_<transformed_column_name>, which you should programatically access instead of repeating all the names in any python code.")
        elif args.task == 'transform':
            if dataset is None:
                print("Dataset key is required for transformation.")
            else:
                transformation = experiment.transform(dataset)
                transformation_csv = transformation.download(dst_file=os.path.join(save_dir, 'transformation.csv'),
                                                             overwrite=True)
                print(f"Transformation saved to {transformation_csv}")
                print(f"Head of transformation:\n{pd.read_csv(transformation_csv).head()}")
        elif args.task in ['fit_transform', 'fit_and_transform']:
            if dataset is None:
                print("Dataset key is required for fit_and_transform.")
            else:
                transformation = experiment.fit_and_transform(dataset)

                if transformation.test_dataset:
                    transformation_csv = transformation.download_transformed_test_dataset(
                        dst_file=os.path.join(save_dir, 'fit_transformation_test.csv'),
                        overwrite=True)
                    print(f"Fit and Transformation on test dataset saved to {transformation_csv}")
                    print(f"Head of fit and transformation on test dataset:\n{pd.read_csv(transformation_csv).head()}")

                if transformation.training_dataset:
                    transformation_csv = transformation.download_transformed_training_dataset(
                        dst_file=os.path.join(save_dir, 'fit_transformation_train.csv'),
                        overwrite=True)
                    print(f"Fit and Transformation on training dataset saved to {transformation_csv}")
                    print(
                        f"Head of fit and transformation on training dataset:\n{pd.read_csv(transformation_csv).head()}")

                if transformation.validation_dataset:
                    print(f"validation_split_fraction: {transformation.validation_split_fraction}")
                    transformation_csv = transformation.download_transformed_validation_dataset(
                        dst_file=os.path.join(save_dir, 'fit_transformation_valid.csv'),
                        overwrite=True)
                    print(f"Fit and Transformation on validation saved to {transformation_csv}")
                    print(
                        f"Head of fit and transformation on validation dataset:\n{pd.read_csv(transformation_csv).head()}")
        elif args.task == 'artifacts':
            get_artifacts(experiment=experiment, save_dir=save_dir)
        elif args.task in ['regression', 'classification']:
            print(f"{args.task} task does not apply when re-using an existing experiment.")
        else:
            print(f"Nothing to do for task {args.task} on experiment {experiment.name}")

    else:
        if args.demo_mode:
            args.data_url = "https://h2o-internal-release.s3-us-west-2.amazonaws.com/data/Splunk/churn.csv"
            args.target_column = "Churn?"
            args.task = "classification"
            args.scorer = "F1"

        # Create and split dataset
        dataset = create_dataset(
            dai,
            args.data_url,
            args.dataset_name,
            args.data_source,
            args.force
        )

        train_test_split = split_dataset(
            dataset,
            args.train_size,
            f"{args.dataset_name}_train",
            f"{args.dataset_name}_test",
            args.target_column,
            args.seed
        )

        # Create and run experiment
        experiment = create_experiment(
            dai,
            train_test_split,
            args.target_column,
            args.scorer,
            args.task,
            args.experiment_name,
            args.accuracy,
            args.time,
            args.interpretability,
            args.force,
            args.fast,
        )

        # Print details and visualize results
        print_experiment_details(experiment)
        visualize_importance(experiment)

        # Individual plots
        metric_plots = experiment.metric_plots
        if args.task == 'classification':
            plot_roc_curve(metric_plots.roc_curve, save_dir)
            plot_precision_recall(metric_plots.prec_recall_curve, save_dir)
            plot_gains_chart(metric_plots.gains_chart, save_dir)
            plot_lift_chart(metric_plots.lift_chart, save_dir)
            plot_ks_chart(metric_plots.ks_chart, save_dir)

            # All plots in one figure
            plot_all_charts(metric_plots.roc_curve, metric_plots.prec_recall_curve, metric_plots.gains_chart,
                            metric_plots.lift_chart, metric_plots.ks_chart, save_dir)
        else:
            # FIXME: Add regression metrics plots
            print("Regression task detected. No classification metrics to plot.")

        get_artifacts(experiment=experiment, save_dir=save_dir)


if __name__ == "__main__":
    main()
