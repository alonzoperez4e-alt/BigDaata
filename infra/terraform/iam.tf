# Roles y usuarios IAM bajo principio de minimo privilegio

resource "aws_iam_role" "glue_role" {
  name = "AWSGlueServiceRole-COES"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "GlueAssumeRole"
        Effect    = "Allow"
        Principal = { Service = "glue.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service_managed" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3_least_privilege" {
  name = "coes-s3-least-privilege"
  role = aws_iam_role.glue_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucketDataLake"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
        ]
        Resource = aws_s3_bucket.datalake.arn
      },
      {
        Sid    = "ReadWriteObjectsDataLake"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts",
        ]
        Resource = "${aws_s3_bucket.datalake.arn}/*"
      }
    ]
  })
}

resource "aws_iam_user" "dashboard_reader" {
  name = "coes-dashboard-reader"
}

resource "aws_iam_user_policy" "dashboard_serving_readonly" {
  name = "coes-serving-readonly"
  user = aws_iam_user.dashboard_reader.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListServingStarSchemaPrefix"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.datalake.arn
        Condition = {
          StringLike = {
            "s3:prefix" = ["serving/star_schema/*"]
          }
        }
      },
      {
        Sid      = "ReadServingStarSchemaObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.datalake.arn}/serving/star_schema/*"
      }
    ]
  })
}

resource "aws_iam_access_key" "dashboard_reader" {
  user = aws_iam_user.dashboard_reader.name
}
