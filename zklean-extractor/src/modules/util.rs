use std::{ffi::OsString, fs::{read_dir, File}, io::Read, path::PathBuf};
use build_fs_tree::{file, BuildError, FileSystemTree, serde_yaml};

pub type FSPath = String;
pub type FSContent = Vec<u8>;
pub type FSTree = FileSystemTree<FSPath, FSContent>;
pub type FSBuildError = BuildError<PathBuf, std::io::Error>;

#[derive(Debug)]
pub enum FSError {
    TemplateError(String),
    BadFilename(OsString),
    BuildError(FSBuildError),
    IOError(std::io::Error),
    DeserializationError(serde_yaml::Error)
}

impl std::error::Error for FSError {}

impl From<std::io::Error> for FSError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}

impl From<FSError> for std::io::Error {
    fn from(value: FSError) -> Self {
        match value {
            FSError::IOError(io_error) => io_error,
            e => std::io::Error::new(std::io::ErrorKind::Other, e),
        }
    }
}

impl From<OsString> for FSError {
    fn from(value: OsString) -> Self {
        Self::BadFilename(value)
    }
}

impl From<FSBuildError> for FSError {
    fn from(value: FSBuildError) -> Self {
        Self::BuildError(value)
    }
}

impl From<serde_yaml::Error> for FSError {
    fn from(value: serde_yaml::Error) -> Self {
        Self::DeserializationError(value)
    }
}

impl std::fmt::Display for FSError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FSError::TemplateError(s) =>
                f.write_fmt(format_args!("Template directory error: {}", s)),
            FSError::BadFilename(s) =>
                f.write_fmt(format_args!("Bad file name in template directory: {:?}", s)),
            FSError::BuildError(build_error) =>
                f.write_fmt(format_args!("Filesystem build error: {}", build_error)),
            FSError::IOError(error) =>
                f.write_fmt(format_args!("IO error: {}", error)),
            FSError::DeserializationError(error) =>
                f.write_fmt(format_args!("bincode decode error: {}", error)),
        }
    }
}

pub type FSResult<T> = Result<T, FSError>;

pub fn read_fs_tree_recursively(root: &PathBuf) -> FSResult<FSTree> {
    if !root.is_dir() {
        Err(FSError::TemplateError("template is not a directory".to_string()))?;
    }

    Ok(FileSystemTree::Directory(
            read_dir(&root)?
            .map(|ent| {
                let ent = ent?;
                let ftype = ent.file_type()?;
                let fname = ent.file_name().into_string()?;

                if ftype.is_file() {
                    let fcontents = {
                        let mut buf: Vec<u8> = vec![];
                        let mut f = File::open(ent.path())?;
                        f.read_to_end(&mut buf)?;
                        buf
                    };

                    Ok((fname, file!(fcontents)))
                } else if ftype.is_dir() {
                    Ok((fname, read_fs_tree_recursively(&ent.path())?))
                } else {
                    Err(FSError::TemplateError("unsupported file type (e.g., symlink)".to_string()))
                }
            }).collect::<FSResult<_>>()?
    ))
}

