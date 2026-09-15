"""CDK stack for the Azul game webapp deployment."""

from __future__ import annotations

import pathlib

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as apigwv2_int,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_cloudwatch as cloudwatch,
    aws_logs as logs,
    aws_ecr_assets as ecr_assets,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class AzulStack(Stack):
    """AWS infrastructure for the Azul game webapp."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        release_id = str(self.node.try_get_context("release_id") or "development")
        live_version = self.node.try_get_context("live_version")
        live_frontend = str(self.node.try_get_context("live_frontend_release") or "pending")

        # --- DynamoDB Tables ---

        games_table = dynamodb.Table(
            self,
            "GamesTable",
            partition_key=dynamodb.Attribute(
                name="game_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
        )
        games_table.add_global_secondary_index(
            index_name="user_sub-updated_at-index",
            partition_key=dynamodb.Attribute(
                name="user_sub", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="updated_at", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        users_table = dynamodb.Table(
            self,
            "UsersTable",
            partition_key=dynamodb.Attribute(
                name="username", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
        )

        # --- S3 Buckets ---

        frontend_bucket = s3.Bucket(
            self,
            "FrontendBucket",
            removal_policy=RemovalPolicy.RETAIN,
            versioned=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )
        release_bucket = s3.Bucket(
            self, "ReleaseBucket", removal_policy=RemovalPolicy.RETAIN,
            versioned=True, enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )
        api_logs = logs.LogGroup(self, "ApiLogs", retention=logs.RetentionDays.ONE_MONTH,
                                 removal_policy=RemovalPolicy.RETAIN)

        # --- Lambda Function (Docker image for PyTorch support) ---

        api_function = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                str(_PROJECT_ROOT),
                file="infra/lambda.Dockerfile",
                platform=ecr_assets.Platform.LINUX_AMD64,
                exclude=[
                    "webapp/node_modules",
                    ".venv",
                    ".git",
                    ".pytest_cache",
                    "cdk.out",
                    "*.egg-info",
                    "agent/runs",
                    "native/astra/target",
                    "play/play_data", ".releases", ".hosting-checks",
                    "webapp", "agent/tests", "play/tests", "__pycache__", "**/__pycache__",
                ],
            ),
            memory_size=2048,
            architecture=lambda_.Architecture.X86_64,
            timeout=Duration.seconds(28),
            log_group=api_logs,
            current_version_options=lambda_.VersionOptions(removal_policy=RemovalPolicy.RETAIN),
            environment={
                "GAMES_TABLE": games_table.table_name,
                "USERS_TABLE": users_table.table_name,
                "AZUL_REQUIRE_MODEL": "1",
                "AZUL_MODEL_REGISTRY": "/var/task/play/artifacts/registry.json",
                "AZUL_TORCH_THREADS": "1",
                "AZUL_RELEASE_ID": release_id,
            },
        )

        games_table.grant_read_write_data(api_function)
        users_table.grant_read_write_data(api_function)
        candidate = api_function.current_version
        serving_version = (lambda_.Version.from_version_attributes(
            self, "ServingVersion", lambda_=api_function, version=str(live_version)
        ) if live_version else candidate)
        live_alias = lambda_.Alias(self, "LiveAlias", alias_name="live", version=serving_version)

        for name, metric, threshold in (
            ("Errors", live_alias.metric_errors(period=Duration.minutes(5)), 1),
            ("Throttles", live_alias.metric_throttles(period=Duration.minutes(5)), 1),
            ("Duration", live_alias.metric_duration(period=Duration.minutes(5), statistic="p95"), 20000),
        ):
            cloudwatch.Alarm(self, f"Api{name}Alarm", metric=metric, threshold=threshold,
                             evaluation_periods=1, treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING)

        # --- API Gateway HTTP API ---

        http_api = apigwv2.HttpApi(
            self,
            "HttpApi",
            api_name="AzulApi",
        )
        cloudwatch.Alarm(
            self, "Http5xxAlarm",
            metric=cloudwatch.Metric(namespace="AWS/ApiGateway", metric_name="5xx",
                dimensions_map={"ApiId": http_api.http_api_id, "Stage": "$default"},
                period=Duration.minutes(5), statistic="Sum"),
            threshold=1, evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        lambda_integration = apigwv2_int.HttpLambdaIntegration(
            "LambdaIntegration", handler=live_alias,
            timeout=Duration.seconds(29),
        )

        # A first deployment provisions a candidate only. Public routes are
        # enabled after the candidate's smoke and latency checks have passed.
        if live_version:
            http_api.add_routes(
                path="/api/{proxy+}", methods=[apigwv2.HttpMethod.ANY],
                integration=lambda_integration,
            )

        # --- CloudFront Distribution ---

        api_origin = origins.HttpOrigin(
            f"{http_api.http_api_id}.execute-api.{self.region}.amazonaws.com",
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        )

        s3_origin = origins.S3BucketOrigin.with_origin_access_control(
            frontend_bucket, origin_path=f"/releases/{live_frontend}",
        )

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=s3_origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
            default_root_object="index.html",
        )

        # --- Frontend Deployment ---

        webapp_dist_path = str(_PROJECT_ROOT / "webapp" / "dist")

        s3deploy.BucketDeployment(
            self,
            "FrontendDeployment",
            sources=[s3deploy.Source.asset(webapp_dist_path)],
            destination_bucket=frontend_bucket,
            destination_key_prefix=f"releases/{release_id}",
            prune=False,
            cache_control=[s3deploy.CacheControl.no_cache()],
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # --- Outputs ---

        CfnOutput(
            self,
            "ApiEndpointUrl",
            value=http_api.url or "",
            description="API Gateway endpoint URL",
        )

        CfnOutput(
            self,
            "CloudFrontUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="CloudFront distribution URL",
        )
        for name, value in {
            "FunctionName": api_function.function_name,
            "CandidateVersion": candidate.version,
            "LiveVersion": str(live_version or "pending"),
            "FrontendRelease": live_frontend,
            "DeploymentRelease": release_id,
            "FrontendBucketName": frontend_bucket.bucket_name,
            "ReleaseBucketName": release_bucket.bucket_name,
            "GamesTableName": games_table.table_name,
            "UsersTableName": users_table.table_name,
            "DistributionId": distribution.distribution_id,
        }.items():
            CfnOutput(self, name, value=value)
