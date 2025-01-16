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
use lance_core::datatypes::Schema as LanceSchema;
use lance_index::DatasetIndexExt;
use lance_index::{IndexParams, IndexType};
use lance_io::object_store::ObjectStoreRegistry;
use std::collections::HashMap;
use std::iter::empty;
use std::sync::Arc;

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

#[derive(Clone)]
pub struct BlockingFile {
    pub(crate) inner: Dataset,
}

impl BlockingFile {
    pub fn drop(uri: &str, storage_options: HashMap<String, String>) -> Result<()> {
        RT.block_on(async move {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let object_store_params = ObjectStoreParams {
                storage_options: Some(storage_options.clone()),
                ..Default::default()
            };
            let (object_store, path) =
                ObjectStore::from_uri_and_params(registry, uri, &object_store_params)
                    .await
                    .map_err(|e| Error::io_error(e.to_string()))?;
            object_store
                .remove_dir_all(path)
                .await
                .map_err(|e| Error::io_error(e.to_string()))
        })
    }
    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = RT.block_on(Dataset::write(reader, uri, params))?;
        Ok(Self { inner })
    }

    pub fn open(
        uri: &str,
        version: Option<i32>,
        block_size: Option<i32>,
        index_cache_size: i32,
        metadata_cache_size: i32,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        let params = ReadParams {
            index_cache_size: index_cache_size as usize,
            metadata_cache_size: metadata_cache_size as usize,
            store_options: Some(ObjectStoreParams {
                block_size: block_size.map(|size| size as usize),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut builder = DatasetBuilder::from_uri(uri).with_read_params(params);

        if let Some(ver) = version {
            builder = builder.with_version(ver as u64);
        }
        builder = builder.with_storage_options(storage_options);

        let inner = RT.block_on(builder.load())?;
        Ok(Self { inner })
    }

    pub fn commit(
        uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        let object_store_registry = Arc::new(ObjectStoreRegistry::default());
        let inner = RT.block_on(Dataset::commit(
            uri,
            operation,
            read_version,
            Some(ObjectStoreParams {
                storage_options: Some(storage_options),
                ..Default::default()
            }),
            None,
            object_store_registry,
            false, // TODO: support enable_v2_manifest_paths
        ))?;
        Ok(Self { inner })
    }

    pub fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<()> {
        RT.block_on(
            self.inner
                .create_index(columns, index_type, name, params, replace),
        )?;
        Ok(())
    }

    pub fn close(&self) {}
}

///////////////////
// Write Methods //
///////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_createWithFfiSchema<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_with_ffi_schema2(
            &mut env,
            arrow_schema_addr,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            storage_options_obj
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_schema2<'local>(
    env: &mut JNIEnv<'local>,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let schema = Schema::try_from(&c_schema)?;

    let reader = RecordBatchIterator::new(empty(), Arc::new(schema));
    create_dataset2(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        storage_options_obj,
        reader,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_drop<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString<'local>,
    storage_options_obj: JObject<'local>,
) -> JObject<'local> {
    let path_str = ok_or_throw!(env, path.extract(&mut env));
    let storage_options =
        ok_or_throw!(env, extract_storage_options(&mut env, &storage_options_obj));
    ok_or_throw!(env, BlockingFile::drop(&path_str, storage_options));
    JObject::null()
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_createWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_with_ffi_stream(
            &mut env,
            arrow_array_stream_addr,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            storage_options_obj
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,   // Optional<Integer>
    max_rows_per_group: JObject,  // Optional<Integer>
    max_bytes_per_file: JObject,  // Optional<Long>
    mode: JObject,                // Optional<String>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    create_dataset2(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        storage_options_obj,
        reader,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_dataset2<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    max_rows_per_file: JObject,
    max_rows_per_group: JObject,
    max_bytes_per_file: JObject,
    mode: JObject,
    storage_options_obj: JObject,
    reader: impl RecordBatchReader + Send + 'static,
) -> Result<JObject<'local>> {
    let path_str = path.extract(env)?;

    let write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &storage_options_obj,
    )?;

    let dataset = BlockingFile::write(reader, &path_str, Some(write_params))?;
    dataset.into_java(env)
}

impl IntoJava for BlockingFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_dataset(env, self)
    }
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingFile,
) -> Result<JObject<'local>> {
    let j_dataset = create_java_dataset_object2(env)?;
    // This block sets a native Rust object (dataset) as a field in the Java object (j_dataset).
    // Caution: This creates a potential for memory leaks. The Rust object (dataset) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_dataset`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    unsafe { env.set_rust_field(&j_dataset, NATIVE_DATASET, dataset) }?;
    Ok(j_dataset)
}

fn create_java_dataset_object2<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("com/lancedb/lance/Dataset", "()V", &[])?;
    Ok(object)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_commitAppend<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<String>, String is json serialized Fragment
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_append2(
            &mut env,
            path,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_append2<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<String>, String is json serialized Fragment)
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let json_fragments = env.get_strings(&fragments_obj)?;
    let mut fragments: Vec<Fragment> = Vec::new();
    for json_fragment in json_fragments {
        let fragment = Fragment::from_json(&json_fragment)?;
        fragments.push(fragment);
    }
    let op = Operation::Append { fragments };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options: HashMap<String, String> = env.with_local_frame(16, |env| {
        let mut map = HashMap::new();
        let mut iter = jmap.iter(env)?;
        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let value_jstring = JString::from(value);
            let key_string: String = env.get_string(&key_jstring)?.into();
            let value_string: String = env.get_string(&value_jstring)?.into();
            map.insert(key_string, value_string);
        }
        Ok::<_, Error>(map)
    })?;
    let dataset = BlockingFile::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_commitOverwrite<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<String>, String is json serialized Fragment
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_overwrite2(
            &mut env,
            path,
            arrow_schema_addr,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_overwrite2<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<String>, String is json serialized Fragment)
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let json_fragments = env.get_strings(&fragments_obj)?;
    let mut fragments: Vec<Fragment> = Vec::new();
    for json_fragment in json_fragments {
        let fragment = Fragment::from_json(&json_fragment)?;
        fragments.push(fragment);
    }
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let arrow_schema = Schema::try_from(&c_schema)?;
    let schema = LanceSchema::try_from(&arrow_schema)?;

    let op = Operation::Overwrite {
        fragments,
        schema,
        config_upsert_values: None,
    };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options: HashMap<String, String> = env.with_local_frame(16, |env| {
        let mut map = HashMap::new();
        let mut iter = jmap.iter(env)?;
        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let value_jstring = JString::from(value);
            let key_string: String = env.get_string(&key_jstring)?.into();
            let value_string: String = env.get_string(&value_jstring)?.into();
            map.insert(key_string, value_string);
        }
        Ok::<_, Error>(map)
    })?;

    let dataset = BlockingFile::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_releaseNativeDataset(
    mut env: JNIEnv,
    obj: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_native_dataset2(&mut env, obj))
}

fn inner_release_native_dataset2(env: &mut JNIEnv, obj: JObject) -> Result<()> {
    let dataset: BlockingFile = unsafe { env.take_rust_field(obj, NATIVE_DATASET)? };
    dataset.close();
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_nativeCreateIndex(
    mut env: JNIEnv,
    java_dataset: JObject,
    columns_jobj: JObject, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject,   // Optional<String>
    params_jobj: JObject, // IndexParams
    replace_jobj: jboolean,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_index2(
            &mut env,
            java_dataset,
            columns_jobj,
            index_type_code_jobj,
            name_jobj,
            params_jobj,
            replace_jobj
        )
    );
}

fn inner_create_index2(
    env: &mut JNIEnv,
    java_dataset: JObject,
    columns_jobj: JObject, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject,   // Optional<String>
    params_jobj: JObject, // IndexParams
    replace_jobj: jboolean,
) -> Result<()> {
    let columns = env.get_strings(&columns_jobj)?;
    let index_type = IndexType::try_from(index_type_code_jobj)?;
    let name = env.get_string_opt(&name_jobj)?;
    let params = get_index_params(env, params_jobj)?;
    let replace = replace_jobj != 0;
    let columns_slice: Vec<&str> = columns.iter().map(AsRef::as_ref).collect();
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingFile>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.create_index(&columns_slice, index_type, name, params.as_ref(), replace)?;
    Ok(())
}

//////////////////
// Read Methods //
//////////////////
#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    version_obj: JObject,    // Optional<Integer>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size: jint,
    metadata_cache_size: jint,
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_open_native2(
            &mut env,
            path,
            version_obj,
            block_size_obj,
            index_cache_size,
            metadata_cache_size,
            storage_options_obj
        )
    )
}

fn inner_open_native2<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    version_obj: JObject,    // Optional<Integer>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size: jint,
    metadata_cache_size: jint,
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let path_str: String = path.extract(env)?;
    let version = env.get_int_opt(&version_obj)?;
    let block_size = env.get_int_opt(&block_size_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options: HashMap<String, String> = env.with_local_frame(16, |env| {
        let mut map = HashMap::new();
        let mut iter = jmap.iter(env)?;

        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let value_jstring = JString::from(value);
            let key_string: String = env.get_string(&key_jstring)?.into();
            let value_string: String = env.get_string(&value_jstring)?.into();
            map.insert(key_string, value_string);
        }

        Ok::<_, Error>(map)
    })?;
    let dataset = BlockingFile::open(
        &path_str,
        version,
        block_size,
        index_cache_size,
        metadata_cache_size,
        storage_options,
    )?;
    dataset.into_java(env)
}

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_File_nativeAddColumnsBySqlExpressions(
    mut env: JNIEnv,
    java_dataset: JObject,
    sql_expressions: JObject, // SqlExpressions
    batch_size: JObject,      // Optional<Long>
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_sql_expressions2(&mut env, java_dataset, sql_expressions, batch_size)
    )
}

fn inner_add_columns_by_sql_expressions2(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql_expressions: JObject, // SqlExpressions
    batch_size: JObject,      // Optional<Long>
) -> Result<()> {
    let sql_expressions_obj = env
        .get_field(sql_expressions, "sqlExpressions", "Ljava/util/List;")?
        .l()?;

    let sql_expressions_obj_list = env.get_list(&sql_expressions_obj)?;
    let mut expressions: Vec<(String, String)> = Vec::new();

    let mut iterator = sql_expressions_obj_list.iter(env)?;

    while let Some(item) = iterator.next(env)? {
        let name = env
            .call_method(&item, "getName", "()Ljava/lang/String;", &[])?
            .l()?;
        let value = env
            .call_method(&item, "getExpression", "()Ljava/lang/String;", &[])?
            .l()?;
        let key_str: String = env.get_string(&JString::from(name))?.into();
        let value_str: String = env.get_string(&JString::from(value))?.into();
        expressions.push((key_str, value_str));
    }

    let rust_transform = NewColumnTransform::SqlExpressions(expressions);

    let batch_size = if env.call_method(&batch_size, "isPresent", "()Z", &[])?.z()? {
        let batch_size_value = env.get_long_opt(&batch_size)?;
        match batch_size_value {
            Some(value) => Some(
                value
                    .try_into()
                    .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
            ),
            None => None,
        }
    } else {
        None
    };

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingFile>(java_dataset, NATIVE_DATASET) }?;

    RT.block_on(
        dataset_guard
            .inner
            .add_columns(rust_transform, None, batch_size),
    )?;
    Ok(())
}
