# Job batch serverless en AWS Glue 4.0 con timeout y reintentos acotados

resource "aws_glue_job" "pipeline_batch" {
  name              = "coes-pipeline-batch"
  role_arn          = aws_iam_role.glue_role.arn
  glue_version      = "4.0"
  worker_type       = var.glue_worker_type
  number_of_workers = var.glue_number_of_workers
  timeout           = 30
  max_retries       = 1

  command {
    name            = "glueetl"
    script_location = "s3://${var.bucket_name}/scripts/glue_runner.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--TempDir"                          = "s3://${var.bucket_name}/tmp/glue/"
    "--extra-py-files"                   = "s3://${var.bucket_name}/scripts/src.zip"
    "--additional-python-modules"        = "scikit-learn==1.5.2,shap==0.49.1,numba==0.59.1,llvmlite==0.42.0,pyarrow==17.0.0,matplotlib==3.9.2"
    "--BUCKET"                           = var.bucket_name
    "--PROFILE"                          = "distributed_managed"
    "--conf"                             = "spark.hadoop.mapred.output.committer.class=org.apache.hadoop.mapred.FileOutputCommitter"
  }
}
