use std::io;

use ha_ndarray::{Axes, Shape};
use freqfs::{DirLock, FileLoad};
use b_table::TableLock;
use safecast::AsType;

mod schema;

pub use schema::TensorSchema;

const METADATA: &str = "metadata";
const BLOCKS: &str = "blocks";
const INDEX: &str = "index";

pub struct Tensor<FE> {
    blocks: DirLock<FE>,
    index: Option<TableLock<TensorSchema, schema::TensorIndexSchema, b_table::collate::Collator<u64>, FE>>,
    shape: Shape,
    axes: Axes,
    block_size: usize,
}

impl<FE> Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    pub async fn create(
        dir: DirLock<FE>,
        shape: Shape,
        axes: Axes,
        sparse: bool,
    ) -> Result<Self, io::Error> {
        let mut dir_guard = dir.try_write()?;

        // Write metadata
        // TODO: proper serialization of shape/axes. For now just placeholder or we rely on them being passed to load?
        // Usually persistent collections store their own metadata.
        // Let's assume we write a small JSON file.
        // For this step I will skip actual metadata write implementation details and focus on structure.
        
        let blocks_dir = dir_guard.create_dir(BLOCKS.to_string())?;
        
        let index = if sparse {
            let index_dir = dir_guard.create_dir(INDEX.to_string())?;
            let schema = TensorSchema::new();
            let collator = b_table::collate::Collator::default();
            Some(TableLock::create(schema, collator, index_dir)?)
        } else {
            None
        };

        Ok(Self {
            blocks: blocks_dir,
            index,
            shape,
            axes,
            block_size: 4096, // Default block size
        })
    }

    pub async fn load(
        dir: DirLock<FE>,
        shape: Shape, 
        axes: Axes,
    ) -> Result<Self, io::Error> {
        let mut dir_guard = dir.try_write()?;
        
        let blocks_dir = dir_guard.get_or_create_dir(BLOCKS.to_string())?;
        
        let index = if dir_guard.contains(INDEX) {
             let index_dir = dir_guard.get_or_create_dir(INDEX.to_string())?;
             let schema = TensorSchema::new();
             let collator = b_table::collate::Collator::default();
             Some(TableLock::load(schema, collator, index_dir)?)
        } else {
            None
        };

        Ok(Self {
            blocks: blocks_dir,
            index,
            shape,
            axes,
            block_size: 4096, // Default block size
        })
    }

    pub async fn read_value_at(&self, coords: Vec<u64>) -> Result<f32, io::Error> {
        // Validation of coords vs shape
        if coords.len() != self.shape.len() {
             return Err(io::Error::new(io::ErrorKind::InvalidInput, "incorrect number of coordinates"));
        }
        
        // TODO: handle axes permutation within coords? Or assume coords are logical and we map to physical?
        // Tensor structure usually stores axes permutation metadata but data is stored in "canonical" order or permuted order?
        // If axes are just metadata for permutation view, then we should permute coords before access.
        // But for this MVP let's assume coords are already physical or we ignore axes permutation for storage index logic for now.
        
        if let Some(index) = &self.index {
            // Sparse
            let coord = coords[0];
            
            // Calculate dense offset
            let mut offset = 0;
            let mut stride = 1;
            for i in (1..coords.len()).rev() {
                 offset += coords[i] * stride;
                 stride *= self.shape[i] as u64;
            }
            
            let block_offset = offset / self.block_size as u64;
            let offset_in_block = (offset % self.block_size as u64) as usize;
            
            let key = vec![coord, block_offset];
            let index_lock = index.read().await;
            if let Some(row) = index_lock.get_row(&key).await? {
                 // Found block
                 let block_id = row[2]; // value column
                 // Read block
                 let blocks = self.blocks.read().await;
                 if let Some(file) = blocks.get_file(&block_id.to_string()) {
                     let guard = file.read::<Vec<f32>>().await?;
                     if offset_in_block < guard.len() {
                         Ok(guard[offset_in_block])
                     } else {
                         Err(io::Error::new(io::ErrorKind::InvalidData, "block offset out of bounds"))
                     }
                 } else {
                     Err(io::Error::new(io::ErrorKind::NotFound, "block file not found"))
                 }
            } else {
                Ok(0.0) // Zero if not present
            }
        } else {
            // Dense
            // Calculate offset
            let mut offset = 0;
            let mut stride = 1;
            for i in (0..coords.len()).rev() {
                 offset += coords[i] * stride;
                 stride *= self.shape[i] as u64;
            }
             
            let block_offset = offset / self.block_size as u64;
            let offset_in_block = (offset % self.block_size as u64) as usize;
            
            // Block name is just block_offset?
             let blocks = self.blocks.read().await;
             if let Some(file) = blocks.get_file(&block_offset.to_string()) {
                 let guard = file.read::<Vec<f32>>().await?;
                 if offset_in_block < guard.len() {
                     Ok(guard[offset_in_block])
                 } else {
                     Err(io::Error::new(io::ErrorKind::InvalidData, "block offset out of bounds"))
                 }
             } else {
                 // Dense tensor usually initializes all blocks? Or lazy?
                 // If lazy, 0.0
                 Ok(0.0)
             }
        }
    }

    pub async fn write_value_at(&self, coords: Vec<u64>, value: f32) -> Result<(), io::Error> {
        if coords.len() != self.shape.len() {
             return Err(io::Error::new(io::ErrorKind::InvalidInput, "incorrect number of coordinates"));
        }

        if let Some(index) = &self.index {
            // Sparse
            let coord = coords[0];
            
            // Calculate dense offset
            let mut offset = 0;
            let mut stride = 1;
            for i in (1..coords.len()).rev() {
                 offset += coords[i] * stride;
                 stride *= self.shape[i] as u64;
            }
            
            let block_offset = offset / self.block_size as u64;
            let offset_in_block = (offset % self.block_size as u64) as usize;
            
            let key = vec![coord, block_offset];
            
            // Check existence
            let block_id = {
                let index_lock = index.read().await;
                if let Some(row) = index_lock.get_row(&key).await? {
                    Some(row[2])
                } else {
                    None
                }
            };

            if let Some(block_id) = block_id {
                let blocks = self.blocks.read().await;
                if let Some(file) = blocks.get_file(&block_id.to_string()) {
                    let mut guard = file.write::<Vec<f32>>().await?;
                    if offset_in_block < guard.len() {
                        guard[offset_in_block] = value;
                        Ok(())
                    } else {
                        Err(io::Error::new(io::ErrorKind::InvalidData, "block offset out of bounds"))
                    }
                } else {
                     Err(io::Error::new(io::ErrorKind::NotFound, "block file not found"))
                }
            } else if value != 0.0 {
                // Insert new block
                let block_id: u64 = rand::random(); 
                {
                    let mut blocks = self.blocks.write().await;
                    blocks.create_file(block_id.to_string(), vec![0.0f32; self.block_size], 0)?;
                }
                
                // Write value
                {
                    let blocks = self.blocks.read().await;
                    let file = blocks.get_file(&block_id.to_string()).expect("just created");
                    let mut guard = file.write::<Vec<f32>>().await?;
                    guard[offset_in_block] = value;
                }
                
                // Update index
                let mut index_lock = index.write().await;
                index_lock.upsert(key, vec![block_id]).await?;
                Ok(())
            } else {
                Ok(()) // Writing 0 to non-existent block -> do nothing
            }
        } else {
            // Dense
            let mut offset = 0;
            let mut stride = 1;
            for i in (0..coords.len()).rev() {
                 offset += coords[i] * stride;
                 stride *= self.shape[i] as u64;
            }
             
            let block_offset = offset / self.block_size as u64;
            let offset_in_block = (offset % self.block_size as u64) as usize;
            
            // Block name is just block_offset
            let file = {
                let blocks = self.blocks.read().await;
                blocks.get_file(&block_offset.to_string()).cloned()
            };
            
            if let Some(file) = file {
                 let mut guard = file.write::<Vec<f32>>().await?;
                 if offset_in_block < guard.len() {
                     guard[offset_in_block] = value;
                     Ok(())
                 } else {
                     Err(io::Error::new(io::ErrorKind::InvalidData, "block offset out of bounds"))
                 }
            } else {
                // Create block if missing (lazy dense)
                {
                    let mut blocks = self.blocks.write().await;
                    blocks.create_file(block_offset.to_string(), vec![0.0f32; self.block_size], 0)?;
                }
                
                let blocks = self.blocks.read().await;
                let file = blocks.get_file(&block_offset.to_string()).expect("just created");
                let mut guard = file.write::<Vec<f32>>().await?;
                guard[offset_in_block] = value;
                Ok(())
            }
        }
    }

}
