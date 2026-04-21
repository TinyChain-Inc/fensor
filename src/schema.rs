use b_table::{IndexSchema, Schema};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct TensorIndexSchema {
    columns: Vec<String>,
}

impl TensorIndexSchema {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }
}

impl b_table::BTreeSchema for TensorIndexSchema {
    type Error = std::io::Error;
    type Value = u64;

    fn block_size(&self) -> usize {
        4096 // TODO: make configurable?
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    fn order(&self) -> usize {
        16 // TODO
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.len() {
            Ok(key)
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid key length"))
        }
    }
}

impl IndexSchema for TensorIndexSchema {
    type Id = String;

    fn columns(&self) -> &[Self::Id] {
        &self.columns
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct TensorSchema {
    primary: TensorIndexSchema,
    auxiliary: Vec<(String, TensorIndexSchema)>,
}

impl TensorSchema {
    pub fn new() -> Self {
        Self {
            primary: TensorIndexSchema::new(vec!["coord".to_string(), "block_offset".to_string(), "block_id".to_string()]),
            auxiliary: vec![],
        }
    }
}

impl Schema for TensorSchema {
    type Id = String;
    type Error = std::io::Error;
    type Value = u64;
    type Index = TensorIndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.primary.columns()[0..2] // coord, block_offset
    }

    fn values(&self) -> &[Self::Id] {
        &self.primary.columns()[2..] // block_id
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, Self::Index)] {
        &self.auxiliary
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
         if key.len() == 2 {
            Ok(key)
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid key length"))
        }
    }

    fn validate_values(&self, values: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
         if values.len() == 1 {
            Ok(values)
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid values length"))
        }
    }
}
