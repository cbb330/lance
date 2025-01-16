// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::traits::FromJString;
use crate::utils::{extract_storage_options, extract_write_params, get_index_params};
use crate::{traits::IntoJava, RT};
use arrow::array::RecordBatchReader;
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatchIterator;
use jni::objects::{JMap, JString};
use jni::sys::jlong;
use jni::sys::{jboolean, jint};
use jni::{objects::JObject, JNIEnv};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::Operation;
use lance::dataset::{Dataset, NewColumnTransform, ReadParams, WriteParams};
use lance::io::{ObjectStore, ObjectStoreParams};
use lance::table::format::Fragment;
use lance_core::cache::FileMetadataCache;
use lance_core::datatypes::Schema as LanceSchema;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::{
    v2::{
        reader::{
            BufferDescriptor, CachedFileMetadata, FileReader, FileReaderOptions, FileStatistics,
        },
        writer::{FileWriter, FileWriterOptions},
    },
    version::LanceFileVersion,
};
use lance_index::DatasetIndexExt;
use lance_index::{IndexParams, IndexType};
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    ReadBatchParams,
};
use object_store::path::Path;
use std::collections::HashMap;
use std::iter::empty;
use std::sync::Arc;
use url::Url;

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

pub struct BlockingFile {
    pub(crate) inner: Arc<FileReader>,
}

// def __init__(self, path: str, storage_options: Optional[Dict[str, str]] = None):
// def read_all(self, *, batch_size: int = 1024, batch_readahead=16) -> ReaderResults:
// def read_range(
// def take_rows(
// def metadata(self) -> LanceFileMetadata:
// def file_statistics(self) -> LanceFileStatistics:
// def read_global_buffer(self, index: int) -> bytes:

impl BlockingFile {
    pub fn open(
        uri_or_path: String,
        storage_options: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        // let inner = RT.block_on(builder.load())?;

        RT.block_on(async move {
            let (object_store, path) =
                object_store_from_uri_or_path(uri_or_path, storage_options).await?;
            let scheduler = ScanScheduler::new(
                Arc::new(object_store),
                SchedulerConfig {
                    io_buffer_size_bytes: 2 * 1024 * 1024 * 1024,
                },
            );
            let file = scheduler
                .open_file(&path)
                .await
                .map_err(|e| Error::io_error(e.to_string()))?;
            let inner = FileReader::try_open(
                file,
                None,
                Arc::<DecoderPlugins>::default(),
                &FileMetadataCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await
            .map_err(|e| Error::io_error(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(inner),
            })
        })
    }
}

// The ObjectStore::from_uri_or_path expects a path to a directory (and it creates it if it does
// not exist).  We are given a path to a file and so we need to strip the last component
// before creating the object store.  We then return the object store and the new relative path
// to the file.
//     pub fn list_indexes(&self) -> Result<Arc<Vec<Index>>> {

pub async fn object_store_from_uri_or_path(
    uri_or_path: impl AsRef<str>,
    storage_options: Option<HashMap<String, String>>,
) -> Result<(ObjectStore, Path)> {
    if let Ok(mut url) = Url::parse(uri_or_path.as_ref()) {
        if url.scheme().len() > 1 {
            //                    .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,

            let path = object_store::path::Path::parse(url.path()).map_err(|e| {
                Error::input_error(format!("Invalid URL path `{}`: {}", url.path(), e))
            })?;
            let (parent_path, filename) = path_to_parent(&path)?;
            url.set_path(parent_path.as_ref());

            let object_store_registry = Arc::new(lance::io::ObjectStoreRegistry::default());
            let object_store_params =
                storage_options
                    .as_ref()
                    .map(|storage_options| ObjectStoreParams {
                        storage_options: Some(storage_options.clone()),
                        ..Default::default()
                    });

            let (object_store, dir_path) = ObjectStore::from_uri_and_params(
                object_store_registry,
                url.as_str(),
                &object_store_params.unwrap_or_default(),
            )
            .await
            .map_err(|e| Error::io_error(e.to_string()))?;

            let child_path = dir_path.child(filename);
            return Ok((object_store, child_path));
        }
    }
    let path = Path::parse(uri_or_path.as_ref()).map_err(|e| {
        Error::input_error(format!("Invalid path `{}`: {}", uri_or_path.as_ref(), e))
    })?;
    let object_store = ObjectStore::local();
    Ok((object_store, path))
}

fn path_to_parent(path: &Path) -> Result<(Path, String)> {
    let mut parts = path.parts().collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(Error::input_error(format!(
            "Path {} is not a valid path to a file",
            path,
        )));
    }
    let filename = parts.pop().unwrap().as_ref().to_owned();
    Ok((Path::from_iter(parts), filename))
}
