# Variables de entrada para la infraestructura AWS de COES-SEIN Big Data

variable "aws_region" {
  description = "Region AWS donde se despliega el Data Lake y AWS Glue."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Nombre globalmente unico del bucket S3 del Data Lake."
  type        = string
}

variable "budget_limit_usd" {
  description = "Limite mensual de gasto en USD para alertas de AWS Budgets."
  type        = number
  default     = 15
}

variable "alert_email" {
  description = "Correo electronico receptor de alertas de presupuesto."
  type        = string
}

variable "glue_worker_type" {
  description = "Tipo de worker para AWS Glue 4.0."
  type        = string
  default     = "G.1X"
}

variable "glue_number_of_workers" {
  description = "Numero de workers asignados al Job batch en AWS Glue."
  type        = number
  default     = 2
}
