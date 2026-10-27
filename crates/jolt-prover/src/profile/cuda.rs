use super::{BackendKind, ProfileArgs};
use jolt_kernels::cuda::{device_available, device_count, request_devices};
use std::num::NonZeroUsize;

impl ProfileArgs {
    pub(super) fn configure_cuda(&self) {
        assert!(
            self.gpus.is_none() || self.backend == BackendKind::Cuda,
            "--gpus requires --backend cuda"
        );
        if self.backend != BackendKind::Cuda {
            return;
        }
        let devices = self.gpus.unwrap_or(NonZeroUsize::MIN).get();
        request_devices(devices);
        assert!(device_available(), "CUDA device and kernels must load");
        assert_eq!(
            device_count(),
            devices,
            "all requested CUDA devices must be usable"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        profile: ProfileArgs,
    }

    #[test]
    fn profile_accepts_positive_gpu_counts() {
        for count in [1, 2] {
            let args = Cli::parse_from([
                "profile",
                "--name",
                "fibonacci",
                "--backend",
                "cuda",
                "--gpus",
                &count.to_string(),
            ]);
            assert_eq!(args.profile.backend, BackendKind::Cuda);
            assert_eq!(args.profile.gpus, NonZeroUsize::new(count));
        }
        assert!(Cli::try_parse_from([
            "profile",
            "--name",
            "fibonacci",
            "--backend",
            "cuda",
            "--gpus",
            "0",
        ])
        .is_err());
    }

    #[test]
    #[should_panic(expected = "--gpus requires --backend cuda")]
    fn cpu_backend_rejects_gpu_count() {
        let args = Cli::parse_from([
            "profile",
            "--name",
            "fibonacci",
            "--backend",
            "optimized",
            "--gpus",
            "1",
        ]);
        args.profile.configure_cuda();
    }
}
