terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "a100_spot" {
  name         = "a100-spot"
  machine_type = var.machine_type
  zone         = var.zone

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-22-04"
      size  = 200
    }
  }

  attached_disk {
    source      = google_compute_disk.persistent_disk.id
    device_name = "persistent-disk"
  }

  guest_accelerator {
    type  = "nvidia-tesla-a100"
    count = var.gpu_count
  }

  metadata_startup_script = file("startup.sh")

  network_interface {
    network = "default"
    access_config {}
  }
}
