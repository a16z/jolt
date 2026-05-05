use std::path::Path;

use super::super::super::EmitError;
use super::super::support::generated_file_path;
use super::super::types::GeneratedCrate;

impl GeneratedCrate {
    pub fn write_to(&self, output_root: impl AsRef<Path>) -> Result<(), EmitError> {
        let crate_root = output_root.as_ref().join(&self.crate_name);
        for file in &self.files {
            let path = generated_file_path(&crate_root, &file.path)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    EmitError::new(format!(
                        "failed to create generated crate directory `{}`: {error}",
                        parent.display()
                    ))
                })?;
            }
            std::fs::write(&path, &file.source).map_err(|error| {
                EmitError::new(format!(
                    "failed to write generated crate file `{}`: {error}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }
}

pub fn write_generated_crates(
    generated_crates: &[GeneratedCrate],
    output_root: impl AsRef<Path>,
) -> Result<(), EmitError> {
    for generated_crate in generated_crates {
        generated_crate.write_to(output_root.as_ref())?;
    }
    Ok(())
}
