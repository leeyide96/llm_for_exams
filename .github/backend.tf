terraform {
  backend "gcs" {
    bucket = "generator-terraform-state"
    prefix = "terraform/state"
  }
}