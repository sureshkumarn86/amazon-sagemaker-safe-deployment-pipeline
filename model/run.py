import argparse
import json
import os
import sys
import time

import boto3

import sagemaker
from sagemaker.image_uris import retrieve
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.model_monitor.dataset_format import DatasetFormat

import stepfunctions
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow


def get_training_image(region):
    return sagemaker.image_uris.retrieve(region=region, framework="xgboost", version="latest")

def get_dev_config(model_name, job_id, role, image_uri, kms_key_id):
    return {
        "Parameters": {
            "ImageRepoUri": image_uri,
            "ModelName": model_name,
            "TrainJobId": job_id,
            "DeployRoleArn": role,
            "ModelVariant": "dev",
            "KmsKeyId": kms_key_id,
        },
        "Tags": {"mlops:model-name": model_name, "mlops:stage": "dev"},
    }


def get_prd_config(model_name, job_id, role, image_uri, kms_key_id, notification_arn):
    dev_config = get_dev_config(model_name, job_id, role, image_uri, kms_key_id)
    prod_params = {
        "ModelVariant": "prd",
        "ScheduleMetricName": "feature_baseline_drift_total_amount",
        "ScheduleMetricThreshold": str("0.20"),
        "NotificationArn": notification_arn,
    }
    prod_tags = {
        "mlops:stage": "prd",
    }
    return {
        "Parameters": dict(dev_config["Parameters"], **prod_params),
        "Tags": dict(dev_config["Tags"], **prod_tags),
    }


def get_pipeline_execution_id(pipeline_name, codebuild_id):
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_state(name=pipeline_name)
    for stage in response["stageStates"]:
        for action in stage["actionStates"]:
            # Return the matching stage with the same external id
            if (
                "latestExecution" in action
                and "externalExecutionId" in action["latestExecution"]
                and action["latestExecution"]["externalExecutionId"] == codebuild_id
            ):
                return stage["latestExecution"]["pipelineExecutionId"]


def get_pipeline_revisions(pipeline_name, execution_id):
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_execution(
        pipelineName=pipeline_name, pipelineExecutionId=execution_id
    )
    return dict(
        (r["name"], r["revisionId"]) for r in response["pipelineExecution"]["artifactRevisions"]
    )


def main(
    # git_branch,
    codebuild_id,
    pipeline_name,
    model_name,
    deploy_role,
    # sagemaker_role,
    sagemaker_bucket,
    data_dir,
    output_dir,
    ecr_dir,
    kms_key_id,
    notification_arn,
):

    # Get the region
    region = boto3.Session().region_name
    print("region: {}".format(region))

    if ecr_dir:
        # Load the image uri and input data config
        with open(os.path.join(ecr_dir, "imageDetail.json"), "r") as f:
            image_uri = json.load(f)["ImageURI"]
    else:
        # Get the the managed image uri for current region
        image_uri = get_training_image(region)
    print("image uri: {}".format(image_uri))

    # Get the job id and source revisions
    job_id = get_pipeline_execution_id(pipeline_name, codebuild_id)
    revisions = get_pipeline_revisions(pipeline_name, job_id)
    git_commit_id = revisions["ModelSourceOutput"]
    data_verison_id = revisions["DataSourceOutput"]
    print("job id: {}".format(job_id))
    print("git commit: {}".format(git_commit_id))
    print("data version: {}".format(data_verison_id))

    # Set the output Data
    output_data = {
        "ModelOutputUri": "s3://{}/{}".format(sagemaker_bucket, model_name),
        "BaselineOutputUri": f"s3://{sagemaker_bucket}/{model_name}/monitoring/baseline/mlops-{model_name}-pbl-{job_id}",
    }
    print("model output uri: {}".format(output_data["ModelOutputUri"]))

    # Pass these into the training method
    hyperparameters = {}
    if os.path.exists(os.path.join(data_dir, "hyperparameters.json")):
        with open(os.path.join(data_dir, "hyperparameters.json"), "r") as f:
            hyperparameters = json.load(f)
            for i in hyperparameters:
                hyperparameters[i] = str(hyperparameters[i])


    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "deploy-model-dev.json"), "w") as f:
        config = get_dev_config(model_name, job_id, deploy_role, image_uri, kms_key_id)
        json.dump(config, f)
    with open(os.path.join(output_dir, "deploy-model-prd.json"), "w") as f:
        config = get_prd_config(
            model_name, job_id, deploy_role, image_uri, kms_key_id, notification_arn
        )
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--codebuild-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ecr-dir", required=False)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--deploy-role", required=True)
    # parser.add_argument("--sagemaker-role", required=True)
    parser.add_argument("--sagemaker-bucket", required=True)
    parser.add_argument("--kms-key-id", required=True)
    # parser.add_argument("--git-branch", required=True)
    parser.add_argument("--workflow-role-arn", required=True)
    parser.add_argument("--notification-arn", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
