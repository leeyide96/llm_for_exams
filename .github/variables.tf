variable "projectid" {
  description = "GCP Project ID"
  type        = string
}

variable "pinecone" {
  description = "Pinecone API"
  type        = string
  sensitive   = true
}

variable "email" {
  description = "SVC Account Email"
  type        = string
  sensitive   = true
}