"""AWS Lambda handler wrapping the FastAPI application via Mangum.

Mangum translates API Gateway events to ASGI (FastAPI) requests,
allowing the same FastAPI app to run both locally and on Lambda.

Deployment:
    Package as a Docker container image and deploy to Lambda.
    See deploy/template.yaml for the SAM/CloudFormation template.
"""

from mangum import Mangum
from api.app import app

handler = Mangum(app, lifespan="auto")
