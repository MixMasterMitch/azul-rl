from __future__ import annotations

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template
from infra.stack import AzulStack


def template(live: bool) -> Template:
    context = {'release_id': 'test-release'}
    if live:
        context.update(live_version='7', live_frontend_release='test-release')
    app = cdk.App(context=context)
    stack = AzulStack(app, 'TestAzul', env=cdk.Environment(account='111111111111', region='us-west-2'))
    return Template.from_stack(stack)


def test_candidate_retains_data_and_has_no_public_routes():
    result = template(False)
    result.resource_count_is('AWS::ApiGatewayV2::Route', 0)
    result.resource_count_is('AWS::Lambda::Alias', 1)
    result.has_resource_properties('AWS::Lambda::Function', {
        'PackageType': 'Image', 'Architectures': ['x86_64'], 'MemorySize': 2048, 'Timeout': 28,
        'Environment': {'Variables': Match.object_like({'AZUL_REQUIRE_MODEL':'1','AZUL_TORCH_THREADS':'1'})},
    })
    for resource in result.find_resources('AWS::DynamoDB::Table').values():
        assert resource['DeletionPolicy'] == 'Retain'
        assert resource['Properties']['PointInTimeRecoverySpecification']['PointInTimeRecoveryEnabled']
        assert resource['Properties']['BillingMode'] == 'PAY_PER_REQUEST'
    for resource in result.find_resources('AWS::S3::Bucket').values():
        assert resource['DeletionPolicy'] == 'Retain'
        assert all(resource['Properties']['PublicAccessBlockConfiguration'].values())
        assert resource['Properties']['VersioningConfiguration']['Status'] == 'Enabled'
    result.has_resource('AWS::Lambda::Version', {'DeletionPolicy':'Retain'})
    result.has_resource_properties('AWS::Logs::LogGroup', {'RetentionInDays': 30})


def test_live_routes_use_alias_and_matching_frontend():
    result = template(True)
    result.has_resource_properties('AWS::Lambda::Alias', {'Name':'live', 'FunctionVersion':'7'})
    result.has_resource_properties('AWS::ApiGatewayV2::Route', {'RouteKey':'ANY /api/{proxy+}'})
    result.has_resource_properties('AWS::CloudFront::Distribution', {
        'DistributionConfig': Match.object_like({
            'Origins': Match.array_with([Match.object_like({'OriginPath':'/releases/test-release'})]),
            'CacheBehaviors': Match.array_with([Match.object_like({
                'PathPattern':'/api/*', 'CachePolicyId':'4135ea2d-6df8-44a3-9df3-4b5a84be39ad',
            })]),
        }),
    })
