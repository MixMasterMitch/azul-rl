"""CDK app entry point."""

from __future__ import annotations

import aws_cdk as cdk

from .stack import AzulStack

app = cdk.App()
AzulStack(app, "AzulStack", env=cdk.Environment(region="us-west-2"))
app.synth()
