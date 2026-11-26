# Salidas de infraestructura y credenciales sensibles de lectura del dashboard

output "bucket_name" {
  description = "Nombre del bucket S3 del Data Lake."
  value       = aws_s3_bucket.datalake.id
}

output "glue_job_name" {
  description = "Nombre del Job batch en AWS Glue."
  value       = aws_glue_job.pipeline_batch.name
}

output "glue_role_arn" {
  description = "ARN del rol IAM ejecutado por AWS Glue."
  value       = aws_iam_role.glue_role.arn
}

output "dashboard_user_name" {
  description = "Usuario IAM de solo lectura para serving/star_schema/."
  value       = aws_iam_user.dashboard_reader.name
}

output "dashboard_access_key_id" {
  description = "Access Key ID del usuario de lectura del dashboard."
  value       = aws_iam_access_key.dashboard_reader.id
  sensitive   = true
}

output "dashboard_secret_access_key" {
  description = "Secret Access Key del usuario de lectura del dashboard."
  value       = aws_iam_access_key.dashboard_reader.secret
  sensitive   = true
}
