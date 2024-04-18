use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

pub trait Serializable {
    fn serialize_to_file(&self, path: &Path) -> Result<(), std::io::Error>;
    fn deserialize_from_file(path: &Path) -> Result<Self, std::io::Error>
    where
        Self: Sized;
}

impl<T> Serializable for T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    fn serialize_to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        // Create file if doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;

        let serialized = serde_json::to_string(self)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    fn deserialize_from_file(path: &Path) -> Result<Self, std::io::Error>
    where
        Self: Sized,
    {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let rv_rows: T = serde_json::from_str(&contents)?;
        Ok(rv_rows)
    }
}
