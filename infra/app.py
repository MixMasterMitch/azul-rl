"""CDK app entry point."""

from __future__ import annotations

import aws_cdk as cdk

from .stack import AzulStack

app = cdk.App()
AzulStack(app, app.node.try_get_context("stack_name") or "AzulStack", env=cdk.Environment(
    account=app.node.try_get_context("account"),
    region=app.node.try_get_context("region") or "us-west-2",
))
app.synth()
