//! A Raster value as JSON, for comparing what two runs of a stage produced.

use anyhow::{bail, ensure, Context, Result};
use raster_runtime::RasterValue;
use serde_json::{json, Value};

pub fn value_json(value: &RasterValue) -> Result<Value> {
    Ok(match value {
        RasterValue::Unit => Value::Null,
        RasterValue::Bool(v) => json!(v),
        RasterValue::Int { value, .. } => match u64::try_from(*value) {
            Ok(v) => json!(v),
            Err(_) => json!(i64::try_from(*value)?),
        },
        RasterValue::Str {
            value,
            truncated: false,
        } => json!(value),
        RasterValue::Bytes {
            data,
            truncated: false,
            ..
        } => json!({"bytes": hex::encode(data)}),
        RasterValue::Struct {
            fields,
            truncated: false,
            ..
        } => {
            let mut object = serde_json::Map::new();
            for (key, child) in fields {
                ensure!(
                    object.insert(key.clone(), value_json(child)?).is_none(),
                    "duplicate field {key}"
                );
            }
            Value::Object(object)
        }
        RasterValue::List {
            elements,
            truncated: false,
            ..
        } => Value::Array(elements.iter().map(value_json).collect::<Result<_>>()?),
        RasterValue::Map {
            entries,
            truncated: false,
            ..
        } => {
            let mut object = serde_json::Map::new();
            for (key, child) in entries {
                let key = value_json(key)?
                    .as_str()
                    .context("non-string map key")?
                    .to_string();
                ensure!(
                    object.insert(key.clone(), value_json(child)?).is_none(),
                    "duplicate map key {key}"
                );
            }
            Value::Object(object)
        }
        _ => bail!("unsupported or truncated inference boundary value"),
    })
}
