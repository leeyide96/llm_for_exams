provider "google" {
  project = var.projectid
  region  = "us-central1"
}


resource "google_cloud_run_v2_service" "generator" {
  name     = "academic-generator"
  location = "asia-southeast1"

  template {
    service_account = var.email
    timeout         = "3000s"
    scaling {
      max_instance_count = 2
    }
    containers {
      image = "gcr.io/${var.projectid}/academic-generator"
      ports {
        container_port = "8501"
      }
      env {
        name = "PINECONE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.pinecone.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "PROJECTID"
        value = var.projectid 
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "8192Mi"
        }
      }
    }
  }
}

resource "google_secret_manager_secret" "pinecone" {
  secret_id = "pinecone-api"
  replication {
    user_managed {
      replicas {
        location = "asia-southeast1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "pinecone_secret" {
  secret      = google_secret_manager_secret.pinecone.name
  secret_data = var.pinecone
  enabled     = true
}


output "academic_generator_url" {
  value = google_cloud_run_v2_service.generator.uri
}
