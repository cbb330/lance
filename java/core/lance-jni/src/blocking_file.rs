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
use arrow::array::{RecordBatch, RecordBatchReader, UInt32Array};
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatchIterator;
use futures::stream::StreamExt;
use jni::objects::{JMap, JString};
use jni::sys::jlong;
use jni::sys::{jboolean, jint};
use jni::{objects::JObject, JNIEnv};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::Operation;
use lance::dataset::{Dataset, NewColumnTransform, ReadParams, WriteParams};
use lance::io::{ObjectStore, ObjectStoreParams, RecordBatchStream};
use lance::table::format::Fragment;
use lance_core::cache::FileMetadataCache;
use lance_core::datatypes::Schema as LanceSchema;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
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
use std::{pin::Pin, sync::Arc};
use url::Url;

pub const NATIVE_FILE: &str = "nativeFileHandle";

pub const NATIVE_LANCE_READER_ADAPTER: &str = "nativeLanceReaderAdapterHandle";

pub struct BlockingFileReader {
    pub(crate) inner: Arc<FileReader>,
}

impl IntoJava for BlockingFileReader {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_file(env, self)
    }
}

fn attach_native_file<'local>(
    env: &mut JNIEnv<'local>,
    file: BlockingFileReader,
) -> Result<JObject<'local>> {
    let j_file = create_java_file_object(env)?;
    // This block sets a native Rust object (file) as a field in the Java object (j_file).
    // Caution: This creates a potential for memory leaks. The Rust object (file) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_file`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    unsafe { env.set_rust_field(&j_file, NATIVE_FILE, file) }?;
    Ok(j_file)
}

fn create_java_file_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("com/lancedb/lance/File", "()V", &[])?;
    Ok(object)
}

impl IntoJava for Box<dyn RecordBatchReader + Send> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_reader_adapter(env, self)
    }
}

fn attach_native_reader_adapter<'local>(
    env: &mut JNIEnv<'local>,
    lance_reader_adapter: LanceReaderAdapter,
) -> Result<JObject<'local>> {
    let j_lance_reader_adapter = create_java_lance_reader_adapter_object(env)?;
    // This block sets a native Rust object (file) as a field in the Java object (j_file).
    // Caution: This creates a potential for memory leaks. The Rust object (file) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_file`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    unsafe {
        env.set_rust_field(
            &j_lance_reader_adapter,
            NATIVE_LANCE_READER_ADAPTER,
            lance_reader_adapter,
        )
    }?;
    Ok(j_lance_reader_adapter)
}

fn create_java_lance_reader_adapter_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("com/lancedb/lance/LanceReaderAdapter", "()V", &[])?;
    Ok(object)
}

// def __init__(self, path: str, storage_options: Optional[Dict[str, str]] = None):
// def read_all(self, *, batch_size: int = 1024, batch_readahead=16) -> ReaderResults:
// def read_range(
// def take_rows(
// def metadata(self) -> LanceFileMetadata:
// def file_statistics(self) -> LanceFileStatistics:
// def read_global_buffer(self, index: int) -> bytes:

impl BlockingFileReader {
    fn open(uri_or_path: String, storage_options: Option<HashMap<String, String>>) -> Result<Self> {
        let (object_store, path) = object_store_from_uri_or_path(uri_or_path, storage_options)?;
        let scheduler = ScanScheduler::new(
            Arc::new(object_store),
            SchedulerConfig {
                io_buffer_size_bytes: 2 * 1024 * 1024 * 1024,
            },
        );
        let file = RT.block_on(scheduler.open_file(&path))?;
        let inner = RT.block_on(FileReader::try_open(
            file,
            None,
            Arc::<DecoderPlugins>::default(),
            &FileMetadataCache::no_cache(),
            FileReaderOptions::default(),
        ))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn read_stream(
        &mut self,
        batch_size: u32,
        batch_readahead: u32,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        // read_stream is a synchronous method but it launches tasks and needs to be
        // run in the context of a tokio runtime
        let inner = self.inner.clone();
        let stream = inner
            .read_stream(
                lance_io::ReadBatchParams::RangeFull,
                batch_size,
                batch_readahead,
                FilterExpression::no_filter(),
            )
            .map_err(|e| Error::io_error(e.to_string()))?;

        Ok(Box::new(LanceReaderAdapter(stream)))
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_readStreamNative<'a>(
    mut env: JNIEnv<'a>,
    jfile: JObject,
    batch_size: jint,
    batch_readahead: jint,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        inner_read_stream(&mut env, jfile, batch_size, batch_readahead)
    )
}

fn inner_read_stream<'local>(
    env: &mut JNIEnv<'local>,
    jfile: JObject,
    batch_size: jint,
    batch_readahead: jint,
) -> Result<JObject<'local>> {
    let batch_size_u32 = batch_size as u32;
    let batch_readahead_u32 = batch_readahead as u32;
    let mut file = unsafe { env.get_rust_field::<_, _, BlockingFileReader>(jfile, NATIVE_FILE) }?;
    let lance_reader_adapter = file.read_stream(batch_size_u32, batch_readahead_u32);
    let obj = match lance_reader_adapter {
        Ok(r) => r.as_ref().into_java(env)?,
        Err(_) => JObject::default(),
    };
    Ok(obj)
}

struct LanceReaderAdapter(Pin<Box<dyn RecordBatchStream>>);

impl Iterator for LanceReaderAdapter {
    type Item = std::result::Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = RT.block_on(self.0.next());
        batch.map(|b| b.map_err(|e| e.into()))
    }
}

impl RecordBatchReader for LanceReaderAdapter {
    fn schema(&self) -> std::sync::Arc<arrow_schema::Schema> {
        self.0.schema().clone()
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(env, inner_open_native(&mut env, path, storage_options_obj))
}

fn inner_open_native<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let path_str: String = path.extract(env)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options: Option<HashMap<String, String>> = env.with_local_frame(16, |env| {
        let mut map = HashMap::new();
        let mut iter = jmap.iter(env)?;

        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let value_jstring = JString::from(value);
            let key_string: String = env.get_string(&key_jstring)?.into();
            let value_string: String = env.get_string(&value_jstring)?.into();
            map.insert(key_string, value_string);
        }

        Ok::<_, Error>(if map.is_empty() { None } else { Some(map) })
    })?;
    let file = BlockingFileReader::open(path_str, storage_options)?;
    file.into_java(env)
}

// #[no_mangle]
// pub extern "system" fn Java_com_lancedb_lance_File_readAll<'local>(
//     mut env: JNIEnv<'local>,
//     _obj: JObject,
//     batch_size: jint,
//     batch_readahead: jint,
// ) -> JObject<'local> {
//     ok_or_throw!(
//         env,
//         inner_read_all(
//             &mut env,
//             batch_size,
//             batch_readahead,
//         )
//     )
// }
//
//
// fn inner_read_all<'local>(
//     env: &mut JNIEnv<'local>,
//     batch_size: jint,
//     batch_readahead: jint,
// ) -> Result<JObject<'local>> {
//     let file = BlockingFileReader::read_all(
//         &path_str,
//         storage_options,
//     )?;
//     dataset.into_java(env)
// }

// The ObjectStore::from_uri_or_path expects a path to a directory (and it creates it if it does
// not exist).  We are given a path to a file and so we need to strip the last component
// before creating the object store.  We then return the object store and the new relative path
// to the file.
//     pub fn list_indexes(&self) -> Result<Arc<Vec<Index>>> {

pub fn object_store_from_uri_or_path(
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

            return RT.block_on(async move {
                let (object_store, dir_path) = ObjectStore::from_uri_and_params(
                    object_store_registry,
                    url.as_str(),
                    &object_store_params.unwrap_or_default(),
                )
                .await
                .map_err(|err| Error::io_error(err.to_string()))?;

                let child_path = dir_path.child(filename);
                Ok((object_store, child_path))
            });
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
