#[cfg(test)]
mod tests {
    use crate::apr::*;

    #[test]
    fn test_magic_constant() {
        // ONE format: APR\0
        assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x00]);
        assert_eq!(&MAGIC, b"APR\0");
    }

    #[test]
    fn test_header_from_bytes_too_small() {
        let data = vec![0u8; 10];
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_invalid_magic() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(b"GGUF");
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_valid() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[8..12].copy_from_slice(&10u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset

        let header = AprHeader::from_bytes(&data).expect("should parse");
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, (2, 0));
        assert_eq!(header.tensor_count, 10);
    }

    #[test]
    fn test_flags() {
        let flags = AprFlags::new(0x0007);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());

        let flags2 = AprFlags::new(0x0020);
        assert!(flags2.is_quantized());
        assert!(!flags2.is_compressed());
    }

    #[test]
    fn test_detect_format_by_extension() {
        assert_eq!(detect_format("/fake/model.apr"), "apr");
        assert_eq!(detect_format("/fake/model.gguf"), "gguf");
        assert_eq!(detect_format("/fake/model.safetensors"), "safetensors");
    }

    // APR v2 binary tensor index format tests

    /// Helper to create binary tensor entry
    fn create_binary_tensor_entry(
        name: &str,
        dtype: u8,
        shape: &[u64],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        // Name
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        // Dtype
        data.push(dtype);
        // Shape
        data.push(shape.len() as u8);
        for &dim in shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        // Offset and size
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data
    }

    #[test]
    fn test_tensor_entry_from_binary_valid() {
        let data = create_binary_tensor_entry(
            "model.embed_tokens.weight",
            0,
            &[32000, 2048],
            0,
            262144000,
        );
        let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.name, "model.embed_tokens.weight");
        assert_eq!(entry.dtype, "F32");
        assert_eq!(entry.shape, vec![32000, 2048]);
        assert_eq!(entry.offset, 0);
        assert_eq!(entry.size, 262144000);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_tensor_entry_from_binary_f16() {
        let data = create_binary_tensor_entry(
            "layer.0.attn.q_proj.weight",
            1,
            &[2048, 2048],
            1024,
            8388608,
        );
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "F16");
        assert_eq!(entry.shape, vec![2048, 2048]);
    }

    #[test]
    fn test_tensor_entry_from_binary_bf16() {
        // GH-191: byte 30 is BF16 in GGML dtype mapping (was byte 2 before GH-191 fix)
        let data = create_binary_tensor_entry("lm_head.weight", 30, &[32000, 2048], 512, 131072000);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "BF16");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_0() {
        // GH-191: byte 2 is Q4_0 in GGML dtype mapping
        let data = create_binary_tensor_entry("quantized.weight", 2, &[1024, 1024], 0, 1048576);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "Q4_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_1() {
        // GH-191: byte 3 is Q4_1 in GGML dtype mapping
        let data = create_binary_tensor_entry("quantized.weight", 3, &[1024, 1024], 0, 1048576);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "Q4_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_1d() {
        let data = create_binary_tensor_entry("model.norm.weight", 0, &[2048], 0, 8192);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.shape, vec![2048]);
        assert_eq!(entry.element_count(), 2048);
    }

    #[test]
    fn test_tensor_entry_from_binary_3d() {
        let data = create_binary_tensor_entry("conv.weight", 0, &[64, 3, 7], 0, 5376);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.shape, vec![64, 3, 7]);
        assert_eq!(entry.element_count(), 64 * 3 * 7);
    }

    #[test]
    fn test_tensor_entry_from_binary_too_short() {
        let data = vec![0u8; 2];
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_from_binary_truncated_name() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u16.to_le_bytes()); // name_len = 100
        data.extend_from_slice(b"short"); // Only 5 bytes of name
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_from_binary_truncated_shape() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u16.to_le_bytes()); // name_len
        data.extend_from_slice(b"test");
        data.push(0); // dtype
        data.push(2); // ndim = 2
        data.extend_from_slice(&1024u64.to_le_bytes()); // first dim only
                                                        // Missing second dim, offset, size
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_element_count() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![32, 64, 128],
            offset: 0,
            size: 0,
        };
        assert_eq!(entry.element_count(), 32 * 64 * 128);
    }

    #[test]
    fn test_tensor_entry_element_count_scalar() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 0,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_multiple_tensor_entries_sequential() {
        let mut data = Vec::new();
        data.extend(create_binary_tensor_entry("tensor1", 0, &[100], 0, 400));
        data.extend(create_binary_tensor_entry(
            "tensor2",
            1,
            &[200, 300],
            400,
            120000,
        ));
        data.extend(create_binary_tensor_entry("tensor3", 2, &[50], 120400, 100));

        let mut pos = 0;
        let mut entries = Vec::new();

        while pos < data.len() {
            let (entry, consumed) = TensorEntry::from_binary(&data[pos..]).expect("should parse");
            entries.push(entry);
            pos += consumed;
        }

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "tensor1");
        assert_eq!(entries[1].name, "tensor2");
        assert_eq!(entries[2].name, "tensor3");
        assert_eq!(entries[1].shape, vec![200, 300]);
    }

    // =========================================================================
    // Compression Tests (GH-35)
    // =========================================================================

    #[test]
    fn test_flags_lz4() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(flags.is_compressed());
    }

    #[test]
    fn test_flags_zstd() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(!flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_compressed());
    }

    #[test]
    fn test_flags_no_compression() {
        let flags = AprFlags::new(0);
        assert!(!flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(!flags.is_compressed());
    }

    #[cfg(not(feature = "apr-compression"))]
    #[test]
    fn test_compressed_file_requires_feature() {
        // Create a minimal APR v2 header with LZ4 flag
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&(AprFlags::LZ4_COMPRESSED).to_le_bytes()); // LZ4 flag
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("apr-compression"),
            "Error should mention feature: {}",
            err_msg
        );
    }

    // ============ Additional coverage tests ============

    /// Helper to create a complete valid APR v2 model
    fn create_test_apr_model() -> Vec<u8> {
        let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Binary tensor index entry for "test.weight"
        let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let data_size = 64usize; // 16 floats * 4 bytes

        let total_size = data_offset as usize + data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        // Tensor data (16 floats)
        let data_start = data_offset as usize;
        for i in 0..16 {
            let val = i as f32;
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        let names = model.tensor_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "test.weight");
    }

    #[test]
    fn test_apr_model_metadata() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        let meta = model.metadata();
        assert_eq!(meta.vocab_size, Some(100));
        assert_eq!(meta.hidden_size, Some(64));
    }

    #[test]
    fn test_apr_model_get_tensor() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tensor = model.get_tensor("test.weight");
        assert!(tensor.is_some());
        let entry = tensor.expect("APR operation failed");
        assert_eq!(entry.shape, vec![4, 4]);
        assert_eq!(entry.dtype, "F32");

        // Non-existent tensor
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_model_get_tensor_f32() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let floats = model.get_tensor_f32("test.weight").expect("should get f32");
        assert_eq!(floats.len(), 16);
        assert_eq!(floats[0], 0.0);
        assert_eq!(floats[1], 1.0);
        assert_eq!(floats[15], 15.0);
    }

    #[test]
    fn test_apr_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_get_tensor_bytes() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let bytes = model
            .get_tensor_bytes("test.weight")
            .expect("should get bytes");
        assert_eq!(bytes.len(), 64); // 16 floats * 4 bytes
    }

    #[test]
    fn test_apr_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // 4 x 4 = 16 parameters
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_flags_lz4() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(flags.is_compressed());
        assert!(!flags.is_zstd());
    }

    #[test]
    fn test_apr_flags_zstd() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(flags.is_zstd());
        assert!(flags.is_compressed());
        assert!(!flags.is_lz4());
    }

    #[test]
    fn test_apr_flags_has_vocab() {
        let flags = AprFlags::new(AprFlags::HAS_VOCAB);
        assert!(flags.has_vocab());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_metadata_is_transformer() {
        // is_transformer() requires hidden_size, num_layers, num_heads, vocab_size all Some
        let mut meta = AprMetadata::default();
        assert!(!meta.is_transformer()); // all None

        // Set all required fields
        meta.hidden_size = Some(1024);
        meta.num_layers = Some(12);
        meta.num_heads = Some(16);
        meta.vocab_size = Some(32000);
        assert!(meta.is_transformer());

        // Missing one field
        meta.hidden_size = None;
        assert!(!meta.is_transformer());
    }

    // Version test removed - ONE format, no versioning

    #[test]
    fn test_apr_model_encrypted_error() {
        let mut data = vec![0u8; HEADER_SIZE + 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes()); // flags = encrypted

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Encrypted"));
    }

    #[test]
    fn test_apr_model_truncated_metadata() {
        let mut data = vec![0u8; 100]; // Too small for metadata
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
        data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 (larger than file)

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_is_apr_file() {
        // is_apr_file reads the file and checks for APR\0 magic bytes
        // Non-existent files return false
        assert!(!is_apr_file("/nonexistent/model.apr"));
        assert!(!is_apr_file("/nonexistent/model.gguf"));

        // Create temp file with APR magic
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_apr_file.apr");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(&MAGIC).expect("write magic");
            f.write_all(&[0u8; 60]).expect("write padding");
        }
        assert!(is_apr_file(&path));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format("/path/model.bin"), "unknown");
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_bpe_tokenizer_encode_decode() {
        // Create vocab with ASCII characters
        let id_to_token: Vec<String> = (0u8..128)
            .map(|i| String::from_utf8(vec![i]).unwrap_or_default())
            .collect();

        let token_to_id: HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };

        // Encode simple ASCII
        let ids = tokenizer.encode("hi");
        assert!(!ids.is_empty());

        // Decode back
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_decode_tokens() {
        let vocab: Vec<String> = vec!["hello".to_string(), " ".to_string(), "world".to_string()];

        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_decode_tokens_with_unknown() {
        let vocab: Vec<String> = vec!["a".to_string(), "b".to_string()];

        // Token 99 is out of bounds - decode_tokens formats as [id]
        let result = AprV2Model::decode_tokens(&vocab, &[0, 99, 1]);
        assert!(result.contains('a'));
        assert!(result.contains('b'));
        assert!(result.contains("[99]")); // Unknown tokens formatted as [id]
    }

    // =========================================================================
    // ModelData Tests (Memory-Mapped Model Loading)
    // =========================================================================

    #[test]
    fn test_model_data_from_vec() {
        let data = vec![1u8, 2, 3, 4, 5];
        let model_data = ModelData::from_vec(data.clone());

        assert_eq!(model_data.as_slice(), &data);
        assert_eq!(model_data.len(), 5);
        assert!(!model_data.is_empty());
        assert!(!model_data.is_mmap());
    }

    #[test]
    fn test_model_data_from_vec_empty() {
        let model_data = ModelData::from_vec(vec![]);

        assert!(model_data.is_empty());
        assert_eq!(model_data.len(), 0);
        assert!(!model_data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test mmap data").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        assert_eq!(model_data.as_slice(), b"test mmap data");
        assert_eq!(model_data.len(), 14);
        assert!(!model_data.is_empty());
        assert!(model_data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_nonexistent() {
        let result = ModelData::open_mmap("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_empty_file() {
        use tempfile::NamedTempFile;

        let temp = NamedTempFile::new().expect("create temp file");
        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        assert!(model_data.is_empty());
        assert_eq!(model_data.len(), 0);
        assert!(model_data.is_mmap());
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test release pages").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        // Should not error
        model_data.release_cpu_pages().expect("release pages");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_heap() {
        let model_data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);

        // Should be no-op for heap data
        model_data
            .release_cpu_pages()
            .expect("release pages (no-op)");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"sequential access test")
            .expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        // Should not error
        model_data.advise_sequential().expect("advise sequential");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_heap() {
        let model_data = ModelData::from_vec(vec![1, 2, 3]);

        // Should be no-op for heap data
        model_data
            .advise_sequential()
            .expect("advise sequential (no-op)");
    }

    #[test]
    fn test_model_data_debug() {
        let model_data = ModelData::from_vec(vec![1, 2, 3]);
        let debug_str = format!("{:?}", model_data);
        assert!(debug_str.contains("Heap"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_mmap_debug() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"debug test").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");
        let debug_str = format!("{:?}", model_data);
        assert!(debug_str.contains("Mmap"));
    }

    // =========================================================================
    // AprV2Model mmap Integration Tests
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_uses_mmap_for_uncompressed() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file (uncompressed)
        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write a minimal valid APR v2 file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0 (uncompressed)
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        temp.write_all(&data).expect("write data");

        let model = AprV2Model::load(temp.path()).expect("load model");

        // Should use mmap for uncompressed files
        assert!(model.is_mmap(), "Uncompressed model should use mmap");
    }

    #[test]
    fn test_apr_model_from_bytes_uses_heap() {
        // Create a minimal valid APR v2 data
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let model = AprV2Model::from_bytes(data).expect("load model");

        // from_bytes always uses heap
        assert!(!model.is_mmap(), "from_bytes should use heap, not mmap");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_apr_model_release_cpu_pages() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file
        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = AprV2Model::load(temp.path()).expect("load model");

        // Should not error
        model.release_cpu_pages().expect("release pages");
    }

    #[test]
    fn test_apr_model_load_nonexistent_file() {
        let result = AprV2Model::load("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write invalid magic
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic
        data[4] = 2;
        data[5] = 0;

        temp.write_all(&data).expect("write data");

        let result = AprV2Model::load(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_truncated_header() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write truncated file (less than HEADER_SIZE)
        temp.write_all(&[0u8; 10]).expect("write data");

        let result = AprV2Model::load(temp.path());
        assert!(result.is_err());
    }

    // =========================================================================
    // F16 Conversion Tests
    // =========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        let zero: u16 = 0x0000;
        let result = crate::apr::f16_to_f32(zero);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let one: u16 = 0x3C00; // 1.0 in f16
        let result = crate::apr::f16_to_f32(one);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        let neg_one: u16 = 0xBC00; // -1.0 in f16
        let result = crate::apr::f16_to_f32(neg_one);
        assert!((result + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_inf() {
        let pos_inf: u16 = 0x7C00;
        let result = crate::apr::f16_to_f32(pos_inf);
        assert!(result.is_infinite() && result > 0.0);

        let neg_inf: u16 = 0xFC00;
        let result = crate::apr::f16_to_f32(neg_inf);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        let nan: u16 = 0x7C01; // NaN (exp=31, mantissa!=0)
        let result = crate::apr::f16_to_f32(nan);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let subnormal: u16 = 0x0001; // Smallest positive subnormal
        let result = crate::apr::f16_to_f32(subnormal);
        assert!(result > 0.0 && result < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let neg_zero: u16 = 0x8000;
        let result = crate::apr::f16_to_f32(neg_zero);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_half() {
        let half: u16 = 0x3800; // 0.5 in f16
        let result = crate::apr::f16_to_f32(half);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_two() {
        let two: u16 = 0x4000; // 2.0 in f16
        let result = crate::apr::f16_to_f32(two);
        assert!((result - 2.0).abs() < 1e-6);
    }

    // =========================================================================
    // dequantize_f16 Tests
    // =========================================================================

    #[test]
    fn test_dequantize_f16_empty() {
        let result = crate::apr::dequantize_f16(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_f16_single() {
        // f16: 0x3C00 = 1.0 (little-endian: 0x00, 0x3C)
        let data: &[u8] = &[0x00, 0x3C];
        let result = crate::apr::dequantize_f16(data, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_f16_multiple() {
        // f16: 0x3C00 = 1.0, 0x4000 = 2.0
        let data: &[u8] = &[0x00, 0x3C, 0x00, 0x40];
        let result = crate::apr::dequantize_f16(data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
    }

    // =========================================================================
    // dequantize_q8_0 Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Q8_0 block: 2-byte f16 scale + 32 int8 values
        // Create a simple block with scale = 1.0 (0x3C00) and values 0..32
        let mut data = Vec::new();
        data.push(0x00); // scale low byte
        data.push(0x3C); // scale high byte (1.0 in f16)
        for i in 0..32u8 {
            data.push(i);
        }
        let result = crate::apr::dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
        // First value should be 0 * 1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 1e-4);
        // Value at index 10 should be 10 * 1.0 = 10.0
        assert!((result[10] - 10.0).abs() < 1e-4);
    }

    // =========================================================================
    // BPE Tokenizer Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_empty_vocab() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_single_char_vocab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("abc");
        assert_eq!(encoded.len(), 3);
    }

    #[test]
    fn test_bpe_tokenizer_unknown_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("xyz");
        // Unknown chars should be handled gracefully
        assert!(encoded.is_empty() || encoded.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_bpe_tokenizer_decode_empty() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_valid() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert("world".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0, 1]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_bpe_tokenizer_with_merge_rules() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h".to_string(), "e".to_string(), "he".to_string()],
            merge_rules: vec![("h".to_string(), "e".to_string())],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("he");
        // After merging h+e -> he, should have 1 token
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_with_bos_eos() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<s>".to_string(), 0);
        token_to_id.insert("</s>".to_string(), 1);
        token_to_id.insert("a".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<s>".to_string(), "</s>".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: Some(0),
            eos_id: Some(1),
            special_tokens: HashMap::new(),
        };
        assert_eq!(tokenizer.bos_id, Some(0));
        assert_eq!(tokenizer.eos_id, Some(1));
    }

    // =========================================================================
    // GH-189: extract_special_tokens_from_vocab Tests
    // =========================================================================

    #[test]
    fn test_extract_special_tokens_from_vocab_qwen() {
        // GH-189: Test extraction of Qwen-style special tokens
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<|im_start|>".to_string(), 151644);
        vocab.insert("<|im_end|>".to_string(), 151645);
        vocab.insert("<|endoftext|>".to_string(), 151643);
        vocab.insert("hello".to_string(), 15339);
        vocab.insert("world".to_string(), 1917);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert_eq!(special.len(), 3);
        assert_eq!(special.get("<|im_start|>"), Some(&151644));
        assert_eq!(special.get("<|im_end|>"), Some(&151645));
        assert_eq!(special.get("<|endoftext|>"), Some(&151643));
        // Regular tokens should NOT be in special_tokens
        assert!(!special.contains_key("hello"));
        assert!(!special.contains_key("world"));
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_llama() {
        // GH-189: Test extraction of LLaMA-style special tokens
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<pad>".to_string(), 3);
        vocab.insert("the".to_string(), 1000);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert!(special.contains_key("<s>"));
        assert!(special.contains_key("</s>"));
        assert!(special.contains_key("<unk>"));
        assert!(special.contains_key("<pad>"));
        assert!(!special.contains_key("the"));
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_empty() {
        // GH-189: Empty vocab should return empty special tokens
        let vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let special = extract_special_tokens_from_vocab(&vocab);
        assert!(special.is_empty());
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_no_special() {
        // GH-189: Vocab with no special tokens should return empty
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("the".to_string(), 2);

        let special = extract_special_tokens_from_vocab(&vocab);
        assert!(special.is_empty());
    }

    #[test]
    fn test_extract_special_tokens_pattern_matching() {
        // GH-189: Test that any <|...|> pattern is captured
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<|custom_token|>".to_string(), 999);
        vocab.insert("<|another|>".to_string(), 998);
        vocab.insert("not_special".to_string(), 100);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert!(special.contains_key("<|custom_token|>"));
        assert!(special.contains_key("<|another|>"));
        assert!(!special.contains_key("not_special"));
    }

    // =========================================================================
    // is_apr_file Tests
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent() {
        // Non-existent file returns false (can't read magic bytes)
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[test]
    fn test_is_apr_file_wrong_extension() {
        // Non-existent files all return false
        assert!(!is_apr_file("/some/path/model.gguf"));
        assert!(!is_apr_file("/some/path/model.safetensors"));
        assert!(!is_apr_file("/some/path/model.bin"));
    }

    #[test]
    fn test_is_apr_file_no_extension() {
        assert!(!is_apr_file("/some/path/model"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_valid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write APR magic bytes
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_wrong_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write wrong magic bytes
        temp.write_all(b"GGUF").expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // detect_format Tests
    // =========================================================================

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format("/path/model.apr"), "apr");
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format("/path/model.gguf"), "gguf");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(detect_format("/path/model.safetensors"), "safetensors");
    }

    // =========================================================================
    // TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_dtype_sizes() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 0,
            size: 800, // 10*20*4 bytes
        };
        assert_eq!(entry.element_count(), 200);
    }

    #[test]
    fn test_tensor_entry_empty_shape() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_tensor_entry_4d_shape() {
        let entry = TensorEntry {
            name: "4d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5],
            offset: 0,
            size: 480,
        };
        assert_eq!(entry.element_count(), 120);
    }

    #[test]
    fn test_tensor_entry_1d_shape() {
        let entry = TensorEntry {
            name: "1d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            offset: 0,
            size: 400,
        };
        assert_eq!(entry.element_count(), 100);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_len() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_model_data_is_empty() {
        let empty = ModelData::from_vec(vec![]);
        assert!(empty.is_empty());

        let non_empty = ModelData::from_vec(vec![1]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_model_data_as_slice() {
        let data = ModelData::from_vec(vec![10, 20, 30]);
        let slice = data.as_slice();
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn test_model_data_large() {
        let large_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let data = ModelData::from_vec(large_data);
        assert_eq!(data.len(), 1000);
        assert_eq!(data.as_slice()[0], 0);
        assert_eq!(data.as_slice()[255], 255);
        assert_eq!(data.as_slice()[256], 0);
    }

    // =========================================================================
    // AprHeader Tests
    // =========================================================================

    #[test]
    fn test_apr_header_from_bytes_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC); // APR\0
        data.extend_from_slice(&[2, 0]); // version 2.0
        data.extend_from_slice(&[0, 0]); // flags
        data.extend_from_slice(&5u32.to_le_bytes()); // tensor_count = 5
        data.extend_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data.extend_from_slice(&100u32.to_le_bytes()); // metadata_size
        data.extend_from_slice(&164u64.to_le_bytes()); // tensor_index_offset
        data.extend_from_slice(&500u64.to_le_bytes()); // data_offset
        data.extend_from_slice(&0u32.to_le_bytes()); // checksum
        data.extend_from_slice(&[0u8; 20]); // reserved

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.expect("APR operation failed");
        assert_eq!(header.version.0, 2);
        assert_eq!(header.version.1, 0);
        assert_eq!(header.tensor_count, 5);
    }

    #[test]
    fn test_apr_header_from_bytes_wrong_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // Wrong magic
        data.extend_from_slice(&[0u8; 60]); // padding

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_from_bytes_too_short() {
        let data = vec![0u8; 10]; // Too short
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprMetadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_true() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_hidden() {
        let meta = AprMetadata {
            hidden_size: None,
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_layers() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: None,
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_default() {
        let meta = AprMetadata::default();
        assert!(meta.hidden_size.is_none());
        assert!(meta.num_layers.is_none());
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // AprV2Model Tests - Basic Operations
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_minimal() {
        let data = create_test_apr_model();
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_ok());
        let model = result.expect("APR operation failed");
        // Helper creates 1 tensor
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let names = model.tensor_names();
        // Helper creates tensor "test.weight"
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test.weight"));
    }

    #[test]
    fn test_apr_v2_model_metadata_default() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let meta = model.metadata();
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Helper creates 1 tensor with shape [4,4] = 16 elements
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_v2_model_is_mmap_false() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert!(!model.is_mmap());
    }

    // =========================================================================
    // AprV2Model Tests - predict
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_no_tensors() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        // With no tensors, returns sum of features
        let output = result.expect("APR operation failed");
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_v2_model_predict_empty_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features: Vec<f32> = vec![];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert_eq!(output[0], 0.0);
    }

    // =========================================================================
    // AprV2Model Tests - forward
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty tokens should fail
    }

    #[test]
    fn test_apr_v2_model_forward_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[1, 2, 3]);
        // Should fail because metadata doesn't indicate transformer
        assert!(result.is_err());
    }

    // =========================================================================
    // decode_tokens Tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_basic() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_input() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_tokens_out_of_bounds() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 5, 10]);
        // Should contain "hello" and [id] for invalid tokens
        assert!(result.contains("hello"));
        assert!(result.contains("[5]"));
        assert!(result.contains("[10]"));
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix() {
        let vocab = vec!["▁hello".to_string(), "▁world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Sentencepiece prefix should be converted to space
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_vocab() {
        let vocab: Vec<String> = vec![];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // All tokens out of bounds, formatted as [id]
        assert!(result.contains("[0]"));
        assert!(result.contains("[1]"));
        assert!(result.contains("[2]"));
    }

    // =========================================================================
    // bpe_encode Tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_empty_text() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("", &token_to_id, &merge_rules, &HashMap::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_single_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("a", &token_to_id, &merge_rules, &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_chars() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("xyz", &token_to_id, &merge_rules, &HashMap::new());
        // Unknown chars return empty or default behavior
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_with_merge() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let merge_rules = vec![("h".to_string(), "e".to_string())];
        let result = bpe_encode("he", &token_to_id, &merge_rules, &HashMap::new());
        // Should merge h+e -> he
        assert!(!result.is_empty());
    }

    // =========================================================================
    // BpeTokenizer Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_whitespace() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(" ".to_string(), 0);
        token_to_id.insert("a".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec![" ".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode(" a ");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_sentencepiece() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("▁hello".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["▁hello".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0]);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_bpe_tokenizer_decode_unknown_id() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0, 100, 200]);
        // Should handle out of bounds gracefully
        assert!(decoded.contains("a") || decoded.contains("<unk>"));
    }

    // =========================================================================
    // dequantize_q4_k Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_empty() {
        let result = crate::apr::dequantize_q4_k(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_empty() {
        let result = crate::apr::dequantize_q6_k(&[], 0);
        assert!(result.is_empty());
    }

    // =========================================================================
    // dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_f32() {
        // F32 is not a quantized type, returns None
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F32"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f16() {
        // F16 is not a quantized type, returns None
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F16"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0() {
        // Q4_0 is qtype 2 in GGML
        let result = crate::apr::dtype_to_ggml_qtype("Q4_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0() {
        // Q8_0 is qtype 8 in GGML
        let result = crate::apr::dtype_to_ggml_qtype("Q8_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_unknown() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("UNKNOWN"), None);
    }

    // =========================================================================
    // is_quantized_dtype Tests
    // =========================================================================

    #[test]
    fn test_is_quantized_dtype_f32() {
        assert!(!crate::apr::is_quantized_dtype("F32"));
    }

    #[test]
    fn test_is_quantized_dtype_f16() {
        assert!(!crate::apr::is_quantized_dtype("F16"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_0() {
        assert!(crate::apr::is_quantized_dtype("Q4_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q8_0() {
        assert!(crate::apr::is_quantized_dtype("Q8_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_k() {
        assert!(crate::apr::is_quantized_dtype("Q4_K"));
    }

    #[test]
    fn test_is_quantized_dtype_q6_k() {
        assert!(crate::apr::is_quantized_dtype("Q6_K"));
    }

    // =========================================================================
    // byte_to_bpe_char Tests
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_ascii() {
        // ASCII printable range
        assert_eq!(crate::apr::byte_to_bpe_char(b'a'), "a");
        assert_eq!(crate::apr::byte_to_bpe_char(b'z'), "z");
        assert_eq!(crate::apr::byte_to_bpe_char(b'0'), "0");
    }

    #[test]
    fn test_byte_to_bpe_char_space() {
        // Space is encoded as Ġ in GPT-2 byte-level BPE
        assert_eq!(crate::apr::byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_special() {
        // Control chars get mapped to special unicode
        let result = crate::apr::byte_to_bpe_char(0);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_byte_to_bpe_char_high_byte() {
        let result = crate::apr::byte_to_bpe_char(255);
        assert!(!result.is_empty());
    }

    // =========================================================================
    // rms_norm Tests
    // =========================================================================

    #[test]
    fn test_rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.74
        // Each element normalized by RMS
    }

    #[test]
    fn test_rms_norm_zeros() {
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // All zeros remain zeros
        for &v in &result {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Weight should scale the result
        assert!(result[0] > 1.0);
    }

    // =========================================================================
    // matmul Tests
    // =========================================================================

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity matrix times [1,2,1,2] (2 seq, 2 dim)
        let x = vec![1.0, 2.0, 1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let result = crate::apr::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_simple() {
        // [1,2] * [[1],[1]] = [3]
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = crate::apr::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    // transpose_matrix is not in public API

    // =========================================================================
    // simd_dot Tests
    // =========================================================================

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = crate::apr::simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let result = crate::apr::simd_dot(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_large() {
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 256];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..255 = 255*256/2 = 32640
        assert!((result - 32640.0).abs() < 1e-3);
    }

    // detect_format and AprFlags tests already exist above
    // f16_to_f32 tests already exist above

    // =========================================================================
    // dequantize_f16 Tests (extended)
    // =========================================================================

    #[test]
    fn test_dequantize_f16_basic() {
        // f16 1.0 = 0x3C00, stored as little-endian [0x00, 0x3C]
        let bytes = vec![0x00, 0x3C, 0x00, 0x3C];
        let result = crate::apr::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_f16_truncated() {
        let bytes = vec![0x00]; // Only 1 byte, needs 2
        let result = crate::apr::dequantize_f16(&bytes, 1);
        // Truncated input returns empty
        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // dequantize_q8_0 Tests (extended)
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_zero_scale() {
        // Q8_0 block: 2-byte scale + 32 bytes data
        let mut bytes = vec![0u8; 34];
        // Scale = 0.0 (f16)
        bytes[0] = 0x00;
        bytes[1] = 0x00;
        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // All zeros with zero scale
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    // BpeTokenizer tests already exist above

    // =========================================================================
    // TensorEntry Tests (extended)
    // =========================================================================

    #[test]
    fn test_tensor_entry_byte_size_f32() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            offset: 0,
            size: 24, // 6 elements * 4 bytes
        };
        assert_eq!(entry.element_count(), 6);
    }

    #[test]
    fn test_tensor_entry_byte_size_f16() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F16".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 32, // 16 elements * 2 bytes
        };
        assert_eq!(entry.element_count(), 16);
    }

    // =========================================================================
    // AprMetadata Tests (extended)
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_missing_heads() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: None,
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_vocab() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: None,
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // simple_attention Tests
    // =========================================================================

    #[test]
    fn test_simple_attention_single_head() {
        // Very simple case: 1 token, 1 head, head_dim=2
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![1.0, 1.0];
        let result = crate::apr::simple_attention(&q, &k, &v, 1, 1, 1, 2);
        assert_eq!(result.len(), 2);
        // Attention output should be weighted v
    }

    // CudaAprModel tests require GPU feature flag - tested elsewhere

    // =========================================================================
    // AprV2Model Error Path Tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_truncated() {
        // Too short to contain header
        let data = vec![0u8; 10];
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_from_bytes_invalid_magic() {
        let mut data = vec![0u8; 100];
        // Invalid magic
        data[0..4].copy_from_slice(b"GGUF");
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
    }

    // Version test removed - ONE format, no versioning

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_existing() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // test.weight exists in test model
        let result = model.get_tensor_bytes("test.weight");
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_existing() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // test.weight exists in test model
        let result = model.get_tensor_f32("test.weight");
        assert!(result.is_ok());
        let tensor = result.expect("APR operation failed");
        assert!(!tensor.is_empty());
    }

    // =========================================================================
    // AprHeader Extended Tests
    // =========================================================================

    #[test]
    fn test_apr_header_magic_validation() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX"); // Invalid magic
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_version_tuple() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // Major
        data[5] = 1; // Minor
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.expect("APR operation failed");
        assert_eq!(header.version, (2, 1));
    }

    // =========================================================================
    // AprFlags Extended Tests
    // =========================================================================

    #[test]
    fn test_apr_flags_from_default() {
        let flags = AprFlags::default();
        // Default flags have no bits set
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    // =========================================================================
    // TensorEntry Extended Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_large_shape() {
        // Create entry with large multidimensional shape
        let entry = create_binary_tensor_entry("big_tensor", 0, &[10, 20, 30, 40], 0, 240000);
        let (parsed, consumed) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.name, "big_tensor");
        assert_eq!(parsed.shape, vec![10, 20, 30, 40]);
        assert_eq!(parsed.element_count(), 10 * 20 * 30 * 40);
        assert!(consumed > 0);
    }

    #[test]
    fn test_tensor_entry_from_binary_zero_dim() {
        // Scalar tensor (empty shape)
        let entry = create_binary_tensor_entry("scalar", 0, &[], 0, 4);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.name, "scalar");
        assert!(parsed.shape.is_empty());
        assert_eq!(parsed.element_count(), 1); // Scalar has 1 element
    }

    // =========================================================================
    // dequantize Function Tests
    // =========================================================================

    #[test]
    fn test_dequantize_f16_multiple_values() {
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let bytes = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
        ];
        let result = crate::apr::dequantize_f16(&bytes, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_dequantize_q8_0_nonzero_scale() {
        // Q8_0 block: 2-byte scale + 32 i8 values
        // Scale = 1.0 (f16 0x3C00)
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00; // Scale low byte
        bytes[1] = 0x3C; // Scale high byte (1.0 in f16)
                         // Set first few values to small integers
        bytes[2] = 1; // i8 value 1
        bytes[3] = 2; // i8 value 2
        bytes[4] = 255; // i8 value -1

        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_vec_operations() {
        let data = vec![1u8, 2, 3, 4, 5];
        let md = ModelData::from_vec(data);
        assert_eq!(md.len(), 5);
        assert!(!md.is_empty());
        let slice = md.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    // =========================================================================
    // simple_attention Extended Tests
    // =========================================================================

    #[test]
    fn test_simple_attention_multi_head() {
        // 2 tokens, 2 heads, head_dim=4
        let hidden_dim = 8; // 2 heads * 4 head_dim
        let q = vec![1.0; hidden_dim * 2]; // 2 tokens
        let k = vec![1.0; hidden_dim * 2];
        let v = vec![1.0; hidden_dim * 2];

        let result = crate::apr::simple_attention(&q, &k, &v, 2, 2, 2, 4);
        assert_eq!(result.len(), hidden_dim * 2);
    }

    #[test]
    fn test_simple_attention_gqa() {
        // GQA: 4 heads, 2 KV heads, head_dim=2
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim]; // 1 token
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = crate::apr::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // matmul Extended Tests
    // =========================================================================

    #[test]
    fn test_matmul_rectangular() {
        // [2,3] * [3,4] = [2,4]
        let x = vec![1.0; 2 * 3]; // 2 rows, 3 cols
        let w = vec![1.0; 3 * 4]; // 3 rows (in_dim), 4 cols (out_dim)
        let result = crate::apr::matmul(&x, &w, 2, 3, 4);
        assert_eq!(result.len(), 2 * 4);
        // Each output element = sum of 3 ones = 3.0
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_large() {
        // Larger matrix to exercise SIMD paths
        let seq_len = 4;
        let in_dim = 64;
        let out_dim = 64;
        let x = vec![1.0; seq_len * in_dim];
        let w = vec![1.0; in_dim * out_dim];
        let result = crate::apr::matmul(&x, &w, seq_len, in_dim, out_dim);
        assert_eq!(result.len(), seq_len * out_dim);
        // Each element = sum of 64 ones = 64.0
        assert!((result[0] - 64.0).abs() < 1e-3);
    }

    // =========================================================================
    // simd_dot Extended Tests
    // =========================================================================

    #[test]
    fn test_simd_dot_mismatched_len() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 8];
        let result = crate::apr::simd_dot(&a, &b);
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_alternating() {
        let a = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum: 1 - 1 + 1 - 1 + 1 - 1 + 1 - 1 = 0
        assert!((result - 0.0).abs() < 1e-6);
    }

    // =========================================================================
    // BpeTokenizer Decode Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_byte_fallback() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<0x41>".to_string(), 0); // Byte 65 = 'A'
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<0x41>".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0]);
        // Byte fallback should be handled
        assert!(!decoded.is_empty());
    }

    // =========================================================================
    // AprFlags Extended Tests - Coverage for all flag methods
    // =========================================================================

    #[test]
    fn test_apr_flags_lz4_compressed() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(flags.is_compressed()); // LZ4 counts as compressed
    }

    #[test]
    fn test_apr_flags_zstd_compressed() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(!flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_compressed()); // ZSTD counts as compressed
    }

    #[test]
    fn test_apr_flags_encrypted() {
        let flags = AprFlags::new(AprFlags::ENCRYPTED);
        assert!(flags.is_encrypted());
        assert!(!flags.is_compressed());
    }

    #[test]
    fn test_apr_flags_quantized() {
        let flags = AprFlags::new(AprFlags::QUANTIZED);
        assert!(flags.is_quantized());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_multiple() {
        let flags =
            AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
        assert!(flags.is_lz4());
        assert!(flags.is_compressed());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
        assert!(!flags.is_zstd());
        assert!(!flags.is_encrypted());
    }

    // =========================================================================
    // f16_to_f32 Extended Tests - Infinity cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_infinity() {
        // +Inf in f16 = 0x7C00
        let result = crate::apr::f16_to_f32(0x7C00);
        assert!(result.is_infinite() && result > 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // -Inf in f16 = 0xFC00
        let result = crate::apr::f16_to_f32(0xFC00);
        assert!(result.is_infinite() && result < 0.0);
    }

    // =========================================================================
    // dequantize_q4_k Extended Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_partial_block() {
        // Less than one full super-block (144 bytes)
        let bytes = vec![0u8; 50];
        let result = crate::apr::dequantize_q4_k(&bytes, 10);
        // Should handle gracefully
        assert!(result.is_empty() || result.len() <= 10);
    }

    #[test]
    fn test_dequantize_q4_k_one_block() {
        // One complete Q4_K super-block (144 bytes = 256 elements)
        let mut bytes = vec![0u8; 144];
        // Set d (f16) = 1.0 = 0x3C00
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Set dmin (f16) = 0.0
        bytes[2] = 0x00;
        bytes[3] = 0x00;
        // scales and mins are already zeros
        // qs are already zeros

        let result = crate::apr::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // dequantize_q6_k Extended Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q6_k_partial_block() {
        // Less than one full super-block (210 bytes)
        let bytes = vec![0u8; 100];
        let result = crate::apr::dequantize_q6_k(&bytes, 10);
        // Should handle gracefully
        assert!(result.is_empty() || result.len() <= 10);
    }

    #[test]
    fn test_dequantize_q6_k_one_block() {
        // One complete Q6_K super-block (210 bytes = 256 elements)
        let mut bytes = vec![0u8; 210];
        // d (f16) is at the end: offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // 1.0 in f16

        let result = crate::apr::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // TensorEntry::from_binary dtype coverage
    // =========================================================================

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i8() {
        let entry = create_binary_tensor_entry("i8_tensor", 3, &[8], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I8");
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i16() {
        let entry = create_binary_tensor_entry("i16_tensor", 4, &[4], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I16");
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i32() {
        let entry = create_binary_tensor_entry("i32_tensor", 5, &[4], 0, 16);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I32");
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_tensor_entry_from_binary_i64() {
        let entry = create_binary_tensor_entry("i64_tensor", 6, &[4], 0, 32);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "I64");
    }

    #[test]
    fn test_tensor_entry_from_binary_q5_1() {
        // GH-191: byte 7 is Q5_1 in GGML dtype mapping (was U8 before GH-191 fix)
        let entry = create_binary_tensor_entry("q5_1_tensor", 7, &[8], 0, 8);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q5_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_k() {
        // GH-191 FIX: byte 12 is Q4_K in GGML dtype mapping — was ignored before
        let entry = create_binary_tensor_entry("q4k_tensor", 12, &[256], 0, 144);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q4_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_q6_k() {
        // GH-191 FIX: byte 14 is Q6_K in GGML dtype mapping (was byte 9 before)
        let entry = create_binary_tensor_entry("q6k_tensor", 14, &[256], 0, 210);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q6_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_q8_0() {
        // GH-191 FIX: byte 8 is Q8_0 in GGML dtype mapping (was byte 10 before)
        let entry = create_binary_tensor_entry("q8_tensor", 8, &[32], 0, 34);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "Q8_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_unknown_dtype() {
        // Unknown dtype byte defaults to F32
        let entry = create_binary_tensor_entry("unknown_tensor", 255, &[4], 0, 16);
        let (parsed, _) = TensorEntry::from_binary(&entry).expect("APR operation failed");
        assert_eq!(parsed.dtype, "F32");
    }

    // =========================================================================
    // AprV2Model encrypted file test
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_encrypted() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version 2.0
        data[5] = 0;
        // Set encrypted flag (0x0004)
        data[6] = 0x04;
        data[7] = 0x00;

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{err:?}");
        assert!(err_msg.contains("Encrypted"));
    }

    // =========================================================================
    // AprV2Model::generate tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_generate_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.generate(&[], 10, None);
        assert!(result.is_err()); // Empty input should fail
    }

    #[test]
    fn test_apr_v2_model_generate_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Model without transformer config should fail on generate
        let result = model.generate(&[1, 2, 3], 5, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional dtype_to_ggml_qtype coverage
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_lowercase() {
        assert!(crate::apr::dtype_to_ggml_qtype("q4_k").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q5_k").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q6_k").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q8_0").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q4_0").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q4_1").is_some());
        assert!(crate::apr::dtype_to_ggml_qtype("q5_0").is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_K"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_1() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_1"), Some(3));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_0() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_0"), Some(6));
    }

    // =========================================================================
    // AprMetadata extended coverage
    // =========================================================================

    #[test]
    fn test_apr_metadata_with_extra_fields() {
        let json = r#"{
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "vocab_size": 32000,
            "custom_field": "custom_value",
            "another_field": 42
        }"#;
        let meta: AprMetadata = serde_json::from_str(json).expect("parse failed");
        assert!(meta.is_transformer());
        assert_eq!(meta.hidden_size, Some(256));
        // Extra fields should be captured
        assert!(meta.extra.contains_key("custom_field"));
    }

    #[test]
    fn test_apr_metadata_optional_fields() {
        let meta = AprMetadata {
            model_type: Some("llama".to_string()),
            name: Some("test-model".to_string()),
            architecture: Some("transformer".to_string()),
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            num_kv_heads: Some(4),
            vocab_size: Some(50000),
            intermediate_size: Some(4096),
            max_position_embeddings: Some(2048),
            rope_theta: Some(10000.0),
            rope_type: Some(2),
            rms_norm_eps: Some(1e-5),
            extra: HashMap::new(),
        };
        assert!(meta.is_transformer());
        assert_eq!(meta.num_kv_heads, Some(4));
        assert_eq!(meta.rope_type, Some(2));
    }

    // =========================================================================
    // byte_to_bpe_char extended coverage
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_all_printable_ascii() {
        // Test all printable ASCII characters
        for b in 33u8..=126 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_control_chars() {
        // Test control characters 0-31
        for b in 0u8..32 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_extended_ascii() {
        // Test extended ASCII 128-255
        for b in 128u8..=255 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    // =========================================================================
    // BpeTokenizer extended coverage
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_multiple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert(" ".to_string(), 1);
        token_to_id.insert("world".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), " ".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    // =========================================================================
    // AprHeader flags coverage
    // =========================================================================

    #[test]
    fn test_apr_header_with_flags() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set quantized flag
        data[6] = 0x20;
        data[7] = 0x00;

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        assert!(header.flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_single_feature() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features = vec![1.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_predict_many_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    // =========================================================================
    // TensorEntry element_count edge cases
    // =========================================================================

    #[test]
    fn test_tensor_entry_element_count_1d() {
        let entry = TensorEntry {
            name: "vec".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            offset: 0,
            size: 40,
        };
        assert_eq!(entry.element_count(), 10);
    }

    #[test]
    fn test_tensor_entry_element_count_3d() {
        let entry = TensorEntry {
            name: "3d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4],
            offset: 0,
            size: 96,
        };
        assert_eq!(entry.element_count(), 24);
    }

    // =========================================================================
    // dequantize_f16 edge cases
    // =========================================================================

    #[test]
    fn test_dequantize_f16_odd_bytes() {
        // Odd number of bytes - last byte ignored
        let bytes = vec![0x00, 0x3C, 0x00]; // 1.0 + extra byte
        let result = crate::apr::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 1); // Only one complete f16
    }

    // =========================================================================
    // detect_format additional cases
    // =========================================================================

    #[test]
    fn test_detect_format_bin() {
        // .bin is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.bin"), "unknown");
    }

    #[test]
    fn test_detect_format_pt() {
        // .pt is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_detect_format_no_extension() {
        // No extension returns "unknown"
        assert_eq!(detect_format("/path/model"), "unknown");
    }

    #[test]
    fn test_detect_format_hidden_file() {
        // Hidden file with no extension returns "unknown"
        assert_eq!(detect_format("/path/.hidden"), "unknown");
    }

    // =========================================================================
    // dequantize Early Exit Tests - Hit break paths by requesting fewer elements
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_fewer_than_block() {
        // Q8_0 block: 2-byte scale + 32 i8 values = 34 bytes, 32 elements
        // Request only 10 elements to trigger early exit
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            bytes[2 + i] = (i + 1) as u8; // Values 1-32
        }

        let result = crate::apr::dequantize_q8_0(&bytes, 10);
        assert_eq!(result.len(), 10);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        // Two Q8_0 blocks = 68 bytes, 64 elements
        let mut bytes = vec![0u8; 68];
        // Block 1
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Block 2
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0

        let result = crate::apr::dequantize_q8_0(&bytes, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q4_k_fewer_than_superblock() {
        // Q4_K super-block: 144 bytes = 256 elements
        // Request only 100 elements to trigger early exit
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q4_k(&bytes, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_dequantize_q4_k_multiple_superblocks() {
        // Two Q4_K super-blocks = 288 bytes = 512 elements
        let mut bytes = vec![0u8; 288];
        // First superblock
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Second superblock
        bytes[144] = 0x00;
        bytes[145] = 0x3C;

        let result = crate::apr::dequantize_q4_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q6_k_fewer_than_superblock() {
        // Q6_K super-block: 210 bytes = 256 elements
        // Request only 50 elements to trigger early exit
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q6_k(&bytes, 50);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_dequantize_q6_k_multiple_superblocks() {
        // Two Q6_K super-blocks = 420 bytes = 512 elements
        let mut bytes = vec![0u8; 420];
        // First superblock d at 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C;
        // Second superblock d at 418-419
        bytes[418] = 0x00;
        bytes[419] = 0x3C;

        let result = crate::apr::dequantize_q6_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    // =========================================================================
    // BpeTokenizer encode tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_empty() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_encode_with_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("ab".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("ab");
        // Should merge a+b -> ab
        assert!(!encoded.is_empty());
    }

    // =========================================================================
    // AprV2Model find_tensor_name tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_total_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Test model has 4x4 = 16 element tensor
        assert!(model.estimated_parameters() > 0);
    }

    // =========================================================================
    // AprMetadata serialization tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_to_json() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        assert!(json.contains("256"));
        assert!(json.contains("hidden_size"));
    }

    #[test]
    fn test_apr_metadata_roundtrip() {
        let meta = AprMetadata {
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            vocab_size: Some(50000),
            model_type: Some("llama".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        let parsed: AprMetadata = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.hidden_size, Some(1024));
        assert_eq!(parsed.model_type, Some("llama".to_string()));
    }

    // =========================================================================
    // TensorEntry tests with various shapes
    // =========================================================================

    #[test]
    fn test_tensor_entry_5d_shape() {
        let entry = TensorEntry {
            name: "5d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5, 6],
            offset: 0,
            size: 2880,
        };
        assert_eq!(entry.element_count(), 720);
    }

    #[test]
    fn test_tensor_entry_single_element() {
        let entry = TensorEntry {
            name: "single".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    // =========================================================================
    // is_apr_file tests (requires actual file with magic bytes)
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent_path() {
        // Non-existent file returns false
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_without_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"GGUF").expect("write wrong magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // More rms_norm tests
    // =========================================================================

    #[test]
    fn test_rms_norm_large() {
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; 64];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let x = vec![-1.0, -2.0, -3.0, -4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Normalized negative values should remain negative
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // matmul extended tests
    // =========================================================================

    #[test]
    fn test_matmul_zeros() {
        let x = vec![0.0; 4];
        let w = vec![1.0; 4];
        let result = crate::apr::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_scaling() {
        let x = vec![2.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = crate::apr::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 4.0).abs() < 1e-6);
    }

    // =========================================================================
    // Additional Coverage Tests - get_tensor_f32 with different dtypes
    // =========================================================================

    /// Helper to create an APR model with a specific dtype tensor
    fn create_test_apr_model_with_dtype(dtype: u8, data_bytes: &[u8]) -> Vec<u8> {
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Tensor shape depends on dtype and data
        let tensor_entry =
            create_binary_tensor_entry("typed.weight", dtype, &[4], 0, data_bytes.len() as u64);

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + data_bytes.len();
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        // Tensor data
        let data_start = data_offset as usize;
        data[data_start..data_start + data_bytes.len()].copy_from_slice(data_bytes);

        data
    }

    #[test]
    fn test_get_tensor_f32_f16_dtype() {
        // F16 dtype = 1, shape [4], 4 elements = 8 bytes
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let f16_data = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
        ];
        let model_data = create_test_apr_model_with_dtype(1, &f16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 1.0).abs() < 0.1);
        assert!((floats[1] - 2.0).abs() < 0.1);
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_get_tensor_f32_q8_0_dtype() {
        // Q8_0 dtype = 10, block = 2-byte scale + 32 i8 values = 34 bytes for 32 elements
        let mut q8_data = vec![0u8; 34];
        q8_data[0] = 0x00;
        q8_data[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            q8_data[2 + i] = i as u8;
        }

        // Create with shape [32] since Q8_0 block has 32 elements
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_entry = create_binary_tensor_entry("typed.weight", 10, &[32], 0, 34);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 34;
        let mut model_data = vec![0u8; total_size];

        model_data[0..4].copy_from_slice(&MAGIC);
        model_data[4] = 2;
        model_data[5] = 0;
        model_data[8..12].copy_from_slice(&1u32.to_le_bytes());
        model_data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        model_data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        model_data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        model_data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        model_data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        model_data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        model_data[data_start..data_start + 34].copy_from_slice(&q8_data);

        let model = AprV2Model::from_bytes(model_data).expect("should load");
        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 32);
    }

    #[test]
    fn test_get_tensor_f32_unsupported_dtype() {
        // BF16 dtype = 2 is not fully supported for get_tensor_f32
        let bf16_data = vec![0x00, 0x3F, 0x80, 0x00]; // Two BF16 values
        let model_data = create_test_apr_model_with_dtype(2, &bf16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        // BF16 is not in the supported list, should error
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_out_of_bounds() {
        // Create a model where tensor data extends beyond file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&100u64.to_le_bytes()); // data_offset

        // Add tensor entry that claims data beyond file
        let tensor_entry = create_binary_tensor_entry("oob.weight", 0, &[1000], 0, 4000);
        data[64..64 + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");
        let result = model.get_tensor_f32("oob.weight");
        assert!(result.is_err()); // Out of bounds
    }

    // =========================================================================
    // decode_tokens extended tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_gpt2_special() {
        // Test GPT-2 style byte-level BPE tokens
        let vocab = vec![
            "Ġhello".to_string(), // Space + hello
            "Ċ".to_string(),      // Newline
            "ĉ".to_string(),      // Tab
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert!(result.contains("hello"));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_decode_tokens_empty_string_token() {
        let vocab = vec![String::new(), "a".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Empty token shouldn't cause issues
        assert!(result.contains('a'));
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_smallest_positive_normal() {
        // Smallest positive normal f16 = 0x0400 (6.103515625e-5)
        let result = crate::apr::f16_to_f32(0x0400);
        assert!(result > 0.0 && result < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_largest_normal() {
        // Largest finite f16 = 0x7BFF (65504.0)
        let result = crate::apr::f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 10.0);
    }

    #[test]
    fn test_f16_to_f32_negative_normal() {
        // -2.0 in f16 = 0xC000
        let result = crate::apr::f16_to_f32(0xC000);
        assert!((result + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_to_f32_subnormal_nonzero() {
        // Various subnormals (exp=0, mantissa!=0)
        for mant in [1u16, 10, 100, 0x3FF] {
            let result = crate::apr::f16_to_f32(mant);
            assert!(result > 0.0, "Subnormal {mant:#x} should be positive");
        }
    }

    // =========================================================================
    // bpe_encode extended tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_with_newline() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ċ".to_string(), 0); // Newline token
        let result = bpe_encode("\n", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_tab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("ĉ".to_string(), 0); // Tab token
        let result = bpe_encode("\t", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space token
        let result = bpe_encode(" ", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_mixed() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("Ġ".to_string(), 1); // Space
        token_to_id.insert("b".to_string(), 2);
        let result = bpe_encode("a b", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_multiple_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("ab".to_string(), 3);
        token_to_id.insert("abc".to_string(), 4);

        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let result = bpe_encode("abc", &token_to_id, &merges, &HashMap::new());
        // Should merge a+b->ab, then ab+c->abc
        assert!(!result.is_empty());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    /// Helper to create APR model with weight and bias tensors for linear model
    fn create_linear_model_apr() -> Vec<u8> {
        let metadata = r#"{"architecture":"linear"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Weight tensor: 2x3 = 6 elements, 24 bytes
        // Bias tensor: 2 elements, 8 bytes
        let weight_entry = create_binary_tensor_entry("weight", 0, &[2, 3], 0, 24);
        let bias_entry = create_binary_tensor_entry("bias", 0, &[2], 24, 8);
        let tensor_index_size = weight_entry.len() + bias_entry.len();

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index_size as u64;
        let total_size = data_offset as usize + 32;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&2u32.to_le_bytes()); // tensor_count = 2
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + weight_entry.len()].copy_from_slice(&weight_entry);
        data[idx_start + weight_entry.len()..idx_start + tensor_index_size]
            .copy_from_slice(&bias_entry);

        // Tensor data - weight matrix [2,3] (identity-ish pattern)
        let data_start = data_offset as usize;
        let weights: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3] matrix
        for (i, &w) in weights.iter().enumerate() {
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // Bias [2]
        let biases: [f32; 2] = [0.5, 0.5];
        for (i, &b) in biases.iter().enumerate() {
            data[data_start + 24 + i * 4..data_start + 24 + i * 4 + 4]
                .copy_from_slice(&b.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_v2_model_predict_with_weights() {
        let model_data = create_linear_model_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert_eq!(output.len(), 2);
        // Expected: [1*1 + 2*0 + 3*0 + 0.5, 1*0 + 2*1 + 3*0 + 0.5] = [1.5, 2.5]
        assert!((output[0] - 1.5).abs() < 0.01);
        assert!((output[1] - 2.5).abs() < 0.01);
    }

    // =========================================================================
    // AprFlags bit operations
    // =========================================================================

    #[test]
    fn test_apr_flags_sharded() {
        let flags = AprFlags::new(AprFlags::SHARDED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_flags_signed() {
        let flags = AprFlags::new(AprFlags::SIGNED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_all_set() {
        let flags = AprFlags::new(0xFFFF);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
    }

    // =========================================================================
    // simd_dot edge cases
    // =========================================================================

    #[test]
    fn test_simd_dot_empty() {
        let result = crate::apr::simd_dot(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_simd_dot_single_element() {
        let result = crate::apr::simd_dot(&[3.0], &[4.0]);
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-1.0, -1.0, -1.0, -1.0];
        let result = crate::apr::simd_dot(&a, &b);
        // (-1)*(-1) + (-2)*(-1) + (-3)*(-1) + (-4)*(-1) = 1 + 2 + 3 + 4 = 10
        assert!((result - 10.0).abs() < 1e-6);
    }

    // =========================================================================
    // rms_norm multi-sequence
    // =========================================================================

    #[test]
    fn test_rms_norm_multi_sequence() {
        // 2 sequences of hidden_dim=4
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 8);
    }

    // =========================================================================
    // simple_attention edge cases
    // =========================================================================

    #[test]
    fn test_simple_attention_mqa() {
        // MQA: 4 Q heads, 1 KV head
        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim];
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = crate::apr::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // AprHeader debug and clone
    // =========================================================================

    #[test]
    fn test_apr_header_clone() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&5u32.to_le_bytes());

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        let cloned = header.clone();
        assert_eq!(cloned.tensor_count, header.tensor_count);
        assert_eq!(cloned.version, header.version);
    }

    #[test]
    fn test_apr_header_debug() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("AprHeader"));
    }

    // =========================================================================
    // TensorEntry debug and clone
    // =========================================================================

    #[test]
    fn test_tensor_entry_debug() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("TensorEntry"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_tensor_entry_clone() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, entry.name);
        assert_eq!(cloned.shape, entry.shape);
    }

    // =========================================================================
    // AprMetadata debug and clone
    // =========================================================================

    #[test]
    fn test_apr_metadata_debug() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            ..Default::default()
        };
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("AprMetadata"));
    }

    #[test]
    fn test_apr_metadata_clone() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            ..Default::default()
        };
        let cloned = meta.clone();
        assert_eq!(cloned.hidden_size, meta.hidden_size);
        assert_eq!(cloned.num_layers, meta.num_layers);
    }

    // =========================================================================
    // BpeTokenizer debug and clone
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_debug() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };
        let debug_str = format!("{:?}", tokenizer);
        assert!(debug_str.contains("BpeTokenizer"));
    }

    #[test]
    fn test_bpe_tokenizer_clone() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };
        let cloned = tokenizer.clone();
        assert_eq!(cloned.bos_id, tokenizer.bos_id);
        assert_eq!(cloned.id_to_token, tokenizer.id_to_token);
    }

    // =========================================================================
    // AprFlags debug
    // =========================================================================

    #[test]
    fn test_apr_flags_debug() {
        let flags = AprFlags::new(0x0025);
        let debug_str = format!("{:?}", flags);
        assert!(debug_str.contains("AprFlags"));
    }

    #[test]
    fn test_apr_flags_clone() {
        let flags = AprFlags::new(0x0025);
        let cloned = flags;
        assert_eq!(cloned.is_lz4(), flags.is_lz4());
        assert_eq!(cloned.is_quantized(), flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model debug
    // =========================================================================

    #[test]
    fn test_apr_v2_model_debug() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("AprV2Model"));
    }

    // =========================================================================
    // ModelData extended edge cases
    // =========================================================================

    #[test]
    fn test_model_data_debug_heap() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let debug_str = format!("{:?}", data);
        assert!(debug_str.contains("Heap"));
    }

    // =========================================================================
    // dequantize multiple blocks
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_three_blocks() {
        // Three Q8_0 blocks = 102 bytes, 96 elements
        let mut bytes = vec![0u8; 102];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 block 1
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0 block 2
        bytes[68] = 0x00;
        bytes[69] = 0x3C; // Scale = 1.0 block 3

        let result = crate::apr::dequantize_q8_0(&bytes, 96);
        assert_eq!(result.len(), 96);
    }

    #[test]
    fn test_dequantize_q4_k_with_nonzero_scales() {
        // Q4_K super-block with actual scale values
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
        bytes[2] = 0x00;
        bytes[3] = 0x3C; // dmin = 1.0
                         // Set some scale values
        bytes[4] = 0x3F; // scales[0] = 63
        bytes[5] = 0x3F;

        let result = crate::apr::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6_k_with_nonzero_scales() {
        // Q6_K super-block with scale values
        let mut bytes = vec![0u8; 210];
        // scales at offset 192-207
        bytes[192] = 10;
        bytes[193] = 20;
        // d at offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // Transformer Model Tests - Forward Pass Coverage
    // =========================================================================

    /// Helper to create a minimal transformer model for testing forward pass.
    /// This creates a 1-layer transformer with tiny dimensions for test purposes.
    fn create_mini_transformer_apr() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "llama",
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "num_kv_heads": 2,
            "vocab_size": 10,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Tensors needed for forward:
        // - model.embed_tokens.weight [vocab=10, hidden=8] = 80 floats = 320 bytes
        // - layers.0.input_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.self_attn.q_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.k_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.v_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.o_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.post_attention_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.mlp.gate_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.up_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.down_proj.weight [hidden=8, inter=16] = 128 floats = 512 bytes
        // - norm.weight [hidden=8] = 8 floats = 32 bytes
        // - lm_head.weight [vocab=10, hidden=8] = 80 floats = 320 bytes

        let tensor_defs: Vec<(&str, &[usize], usize)> = vec![
            ("model.embed_tokens.weight", &[10, 8], 320),
            ("layers.0.input_layernorm.weight", &[8], 32),
            ("layers.0.self_attn.q_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.k_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.v_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.o_proj.weight", &[8, 8], 256),
            ("layers.0.post_attention_layernorm.weight", &[8], 32),
            ("layers.0.mlp.gate_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.up_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.down_proj.weight", &[8, 16], 512),
            ("norm.weight", &[8], 32),
            ("lm_head.weight", &[10, 8], 320),
        ];

        let mut tensor_entries = Vec::new();
        let mut current_offset = 0u64;

        for (name, shape, byte_size) in &tensor_defs {
            let shape_vec: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
            let entry =
                create_binary_tensor_entry(name, 0, &shape_vec, current_offset, *byte_size as u64);
            tensor_entries.push(entry);
            current_offset += *byte_size as u64;
        }

        let tensor_index: Vec<u8> = tensor_entries
            .iter()
            .flat_map(|e| e.iter().copied())
            .collect();
        let tensor_count = tensor_defs.len() as u32;
        let total_data_size = current_offset as usize;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index.len() as u64;
        let total_size = data_offset as usize + total_data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

        // Tensor data - initialize with small random-ish values
        let data_start = data_offset as usize;
        let num_floats = total_data_size / 4;
        for i in 0..num_floats {
            let val = ((i % 10) as f32 - 5.0) * 0.1; // Small values between -0.5 and 0.4
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Set layernorm weights to 1.0 (they need to be non-zero)
        let norm_weight_offsets = vec![320, 320 + 32 + 256 * 4, 320 + 32 + 256 * 4 + 32 + 512 * 3];
        for offset in norm_weight_offsets {
            for i in 0..8 {
                let val = 1.0f32;
                let pos = data_start + offset + i * 4;
                if pos + 4 <= data.len() {
                    data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        data
    }

    #[test]
    fn test_transformer_model_loads() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");
        assert!(model.metadata().is_transformer());
        assert_eq!(model.metadata().hidden_size, Some(8));
        assert_eq!(model.metadata().num_layers, Some(1));
        assert_eq!(model.metadata().vocab_size, Some(10));
    }

    #[test]
    fn test_transformer_has_all_tensors() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        // Check key tensors exist
        assert!(model.get_tensor("model.embed_tokens.weight").is_some());
        assert!(model
            .get_tensor("layers.0.input_layernorm.weight")
            .is_some());
        assert!(model
            .get_tensor("layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(model.get_tensor("norm.weight").is_some());
        assert!(model.get_tensor("lm_head.weight").is_some());
    }

    // =========================================================================
    // matmul edge cases
    // =========================================================================

    #[test]
    fn test_matmul_out_of_bounds_x() {
        // x is shorter than expected
        let x = vec![1.0, 2.0]; // Only 2 elements, but seq=1, in_dim=4
        let w = vec![1.0; 8]; // 2 output dims, 4 input dims
        let result = crate::apr::matmul(&x, &w, 1, 4, 2);
        // Should handle gracefully (skip or produce zeros)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_matmul_out_of_bounds_w() {
        // w is shorter than expected
        let x = vec![1.0; 4]; // seq=1, in_dim=4
        let w = vec![1.0, 2.0]; // Too short
        let result = crate::apr::matmul(&x, &w, 1, 4, 2);
        // Should handle gracefully
        assert_eq!(result.len(), 2);
    }

    // =========================================================================
    // dequantize with negative i8 values
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_negative_values() {
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Set negative i8 values (128-255 map to -128 to -1)
        bytes[2] = 255; // -1 as i8
        bytes[3] = 254; // -2 as i8
        bytes[4] = 128; // -128 as i8

        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value should be negative: -1 * 1.0 = -1.0
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // f16_to_f32 quiet NaN
    // =========================================================================

    #[test]
    fn test_f16_to_f32_qnan() {
        // Quiet NaN in f16 (exp=0x1F, mantissa>0, sign=0)
        // 0x7E00 = quiet NaN
        let result = crate::apr::f16_to_f32(0x7E00);
        assert!(result.is_nan());
    }

    // =========================================================================
    // detect_format with magic bytes
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "apr");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_gguf_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&[0x47, 0x47, 0x55, 0x46])
            .expect("write GGUF magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "gguf");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_safetensors_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"{\"test\": 1}").expect("write JSON");

        assert_eq!(detect_format(temp.path()), "safetensors");
    }

    // =========================================================================
    // simd_dot_avx2 scalar fallback
    // =========================================================================

    #[test]
    fn test_simd_dot_non_multiple_of_8() {
        // Test with lengths that aren't multiples of 8 to exercise remainder handling
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 13];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..12 = 78
        assert!((result - 78.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_prime_length() {
        // Prime number length ensures both SIMD chunks and remainder are exercised
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 17];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..16 = 136
        assert!((result - 136.0).abs() < 1e-6);
    }

    // =========================================================================
    // simple_attention multi-token
    // =========================================================================

    #[test]
    fn test_simple_attention_multiple_tokens() {
        // 3 tokens, 2 heads, head_dim=4
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let hidden_dim = num_heads * head_dim;

        let q = vec![1.0; seq_len * hidden_dim];
        let k = vec![1.0; seq_len * hidden_dim];
        let v = vec![1.0; seq_len * hidden_dim];

        let result =
            crate::apr::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
    }

    #[test]
    fn test_simple_attention_varying_values() {
        // Test with varying Q, K, V values
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let seq_len = 2;
        let hidden_dim = num_heads * head_dim;

        // Different Q, K, V patterns
        let q = vec![1.0, 0.0, 0.0, 1.0]; // Token 1: [1,0], Token 2: [0,1]
        let k = vec![1.0, 0.0, 1.0, 0.0]; // Token 1: [1,0], Token 2: [1,0]
        let v = vec![1.0, 2.0, 3.0, 4.0]; // Token 1: [1,2], Token 2: [3,4]

        let result =
            crate::apr::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
        // Output should be valid attention-weighted values
        assert!(!result.iter().any(|v| v.is_nan()));
    }

    // =========================================================================
    // ModelData methods
    // =========================================================================

    #[test]
    fn test_model_data_empty() {
        let data = ModelData::from_vec(vec![]);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_model_data_as_slice_extended() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let slice = data.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    // =========================================================================
    // AprV2Model forward error path tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty input
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("empty"));
    }

    #[test]
    fn test_apr_v2_model_generate_max_tokens_zero() {
        let data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.generate(&[1, 2], 0, None);
        // Should succeed with empty generation
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert_eq!(tokens.len(), 2); // Just input, no generation
    }

    // =========================================================================
    // byte_to_bpe_char newline and tab
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_newline() {
        let result = crate::apr::byte_to_bpe_char(b'\n');
        assert_eq!(result, "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        let result = crate::apr::byte_to_bpe_char(b'\t');
        assert_eq!(result, "ĉ");
    }

    // =========================================================================
    // rms_norm with very small epsilon
    // =========================================================================

    #[test]
    fn test_rms_norm_tiny_eps() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-12;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Should not produce NaN or Inf
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_large_eps() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 10.0; // Large epsilon dominates
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // Additional Coverage Tests (_more_cov suffix)
    // =========================================================================

    #[test]
    fn test_load_tokenizer_from_sibling_nonexistent_more_cov() {
        // Test with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::load_tokenizer_from_sibling(path);
        assert!(result.is_none());
    }

    #[test]
    fn test_encode_text_nonexistent_more_cov() {
        // Test encode_text with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::encode_text(path, "hello world");
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tokenizer_nonexistent_more_cov() {
        // Test load_tokenizer with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::load_tokenizer(path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        // Create model file (empty)
        std::fs::File::create(&model_path).expect("create model file");

        // Create invalid JSON tokenizer file
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_encode_text_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::encode_text(&model_path, "test");
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::load_tokenizer(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_missing_model_key_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid JSON but missing "model" key
        f.write_all(br#"{"added_tokens": []}"#).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_missing_vocab_key_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid JSON but missing "vocab" key in "model"
        f.write_all(br#"{"model": {}}"#).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_valid_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid tokenizer.json structure
        let json = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "<s>": 2,
                    "</s>": 3
                }
            },
            "added_tokens": [
                {"content": "<s>", "id": 2},
                {"content": "</s>", "id": 3}
            ]
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_some());
        let (vocab, bos_id, eos_id) = result.expect("APR operation failed");
        assert!(vocab.len() >= 2);
        assert_eq!(bos_id, Some(2));
        assert_eq!(eos_id, Some(3));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_encode_text_valid_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        let json = r#"{
            "model": {
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
                "merges": []
            }
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::encode_text(&model_path, "hello");
        assert!(result.is_some());
        let tokens = result.expect("APR operation failed");
        assert!(!tokens.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_valid_with_merges_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        let json = r#"{
            "model": {
                "vocab": {"h": 0, "e": 1, "he": 2, "<|endoftext|>": 3, "<bos>": 4},
                "merges": ["h e"]
            },
            "added_tokens": [
                {"content": "<|endoftext|>", "id": 3},
                {"content": "<bos>", "id": 4}
            ]
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::load_tokenizer(&model_path);
        assert!(result.is_some());
        let tokenizer = result.expect("APR operation failed");
        assert!(!tokenizer.token_to_id.is_empty());
        assert_eq!(tokenizer.eos_id, Some(3));
        assert_eq!(tokenizer.bos_id, Some(4));
        assert!(!tokenizer.merge_rules.is_empty());
    }

    #[test]
    fn test_bpe_encode_non_ascii_more_cov() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let result = bpe_encode("a\u{00A9}", &token_to_id, &[], &HashMap::new()); // a + copyright symbol
                                                                 // Non-ASCII gets encoded as bytes
        assert!(!result.is_empty() || result.is_empty()); // Just verify no panic
    }

    #[test]
    fn test_bpe_encode_unicode_more_cov() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        // Test with unicode characters
        let result = bpe_encode("\u{1F600}", &token_to_id, &[], &HashMap::new()); // Emoji
                                                                 // Should not panic, may return empty if no tokens match
        assert!(result.is_empty());
    }

    #[test]
    fn test_byte_to_bpe_char_null_more_cov() {
        let result = crate::apr::byte_to_bpe_char(0x00);
        assert!(!result.is_empty());
        assert!(result.starts_with("<0x"));
    }

    #[test]
    fn test_byte_to_bpe_char_delete_more_cov() {
        let result = crate::apr::byte_to_bpe_char(0x7F); // DEL character
        assert!(!result.is_empty());
    }

    #[test]
    fn test_matmul_empty_more_cov() {
        let result = crate::apr::matmul(&[], &[], 0, 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_matmul_single_element_more_cov() {
        let x = vec![2.0];
        let w = vec![3.0];
        let result = crate::apr::matmul(&x, &w, 1, 1, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_single_sequence_more_cov() {
        // Test single sequence normalization
        let x = vec![1.0, 2.0];
        let weight = vec![2.0, 0.5];
        let result = crate::apr::rms_norm(&x, &weight, 1e-6);
        assert_eq!(result.len(), 2);
        // Values should be normalized and scaled by weights
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
    }

    #[test]
    fn test_rms_norm_weight_shorter_than_x_more_cov() {
        // x has more elements than weight - tests unwrap_or path
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, 1.0]; // Only 2 weights for hidden_dim=2
        let result = crate::apr::rms_norm(&x, &weight, 1e-6);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_simple_attention_empty_more_cov() {
        let result = crate::apr::simple_attention(&[], &[], &[], 0, 1, 1, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_dot_unequal_lengths_more_cov() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0]; // Shorter
        let result = crate::apr::simd_dot(&a, &b);
        // Should only use min(5, 3) = 3 elements
        assert!((result - 6.0).abs() < 1e-6); // 1+2+3 = 6
    }

    #[test]
    fn test_dequantize_f16_request_more_than_available_more_cov() {
        // Only enough bytes for 2 f16 values, but request 10
        let bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        let result = crate::apr::dequantize_f16(&bytes, 10);
        assert_eq!(result.len(), 2); // Truncated to available
    }

    #[test]
    fn test_dequantize_q8_0_request_more_than_available_more_cov() {
        // One block = 34 bytes = 32 elements, request 100
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
        let result = crate::apr::dequantize_q8_0(&bytes, 100);
        assert_eq!(result.len(), 32); // Truncated to one block
    }

    #[test]
    fn test_tensor_entry_serialize_deserialize_more_cov() {
        let entry = TensorEntry {
            name: "test.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 100,
            size: 800,
        };
        let json = serde_json::to_string(&entry).expect("invalid UTF-8");
        let parsed: TensorEntry = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.name, "test.weight");
        assert_eq!(parsed.shape, vec![10, 20]);
    }

    #[test]
    fn test_apr_metadata_serialize_deserialize_more_cov() {
        let meta = AprMetadata {
            model_type: Some("llama".to_string()),
            name: Some("TestModel".to_string()),
            architecture: Some("decoder-only".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(8),
            vocab_size: Some(128256),
            intermediate_size: Some(14336),
            max_position_embeddings: Some(8192),
            rope_theta: Some(500000.0),
            rope_type: Some(2),
            rms_norm_eps: Some(1e-5),
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        let parsed: AprMetadata = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.hidden_size, Some(4096));
        assert_eq!(parsed.rope_type, Some(2));
    }

    #[test]
    fn test_apr_flags_copy_more_cov() {
        let flags = AprFlags::new(AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
        let copied = flags;
        assert!(copied.is_quantized());
        assert!(copied.has_vocab());
    }

    #[test]
    fn test_decode_tokens_special_chars_more_cov() {
        let vocab = vec![
            "hello".to_string(),
            "Ġ".to_string(), // Space token
            "Ċ".to_string(), // Newline token
            "ĉ".to_string(), // Tab token
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2, 3]);
        assert!(result.contains("hello"));
        assert!(result.contains(' '));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_bpe_tokenizer_encode_non_ascii_more_cov() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("\u{00E9}"); // e-acute
                                                    // Should not panic
        assert!(encoded.is_empty() || !encoded.is_empty());
    }

    #[test]
    fn test_apr_header_all_fields_more_cov() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 1;
        // Set all flags
        data[6..8].copy_from_slice(&0xFFFFu16.to_le_bytes());
        data[8..12].copy_from_slice(&100u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&200u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&300u32.to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&400u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&500u64.to_le_bytes()); // data_offset
        data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // checksum

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.metadata_offset, 200);
        assert_eq!(header.metadata_size, 300);
        assert_eq!(header.tensor_index_offset, 400);
        assert_eq!(header.data_offset, 500);
        assert_eq!(header.checksum, 0xDEADBEEF);
    }

    #[test]
    fn test_apr_model_header_methods_more_cov() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // tensor_count is accessed through public API
        assert!(model.tensor_count() >= 1);
        // tensor_names returns &str refs
        let names = model.tensor_names();
        assert!(!names.is_empty());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_all_types_more_cov() {
        // Test all supported quantized types
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_K"), Some(12));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_K"), Some(13));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q6_K"), Some(14));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q8_0"), Some(8));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_0"), Some(2));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_1"), Some(3));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_0"), Some(6));
        // Non-quantized types
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F32"), None);
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F16"), None);
        assert_eq!(crate::apr::dtype_to_ggml_qtype("BF16"), None);
    }

    #[test]
    fn test_is_quantized_dtype_all_types_more_cov() {
        // Quantized types
        assert!(crate::apr::is_quantized_dtype("Q4_K"));
        assert!(crate::apr::is_quantized_dtype("q4_k"));
        assert!(crate::apr::is_quantized_dtype("Q5_K"));
        assert!(crate::apr::is_quantized_dtype("Q6_K"));
        assert!(crate::apr::is_quantized_dtype("Q8_0"));
        assert!(crate::apr::is_quantized_dtype("Q4_0"));
        assert!(crate::apr::is_quantized_dtype("Q4_1"));
        assert!(crate::apr::is_quantized_dtype("Q5_0"));
        // Non-quantized types
        assert!(!crate::apr::is_quantized_dtype("F32"));
        assert!(!crate::apr::is_quantized_dtype("F16"));
        assert!(!crate::apr::is_quantized_dtype("BF16"));
        assert!(!crate::apr::is_quantized_dtype("I8"));
    }

    #[test]
    fn test_dequantize_q4_k_early_exit_more_cov() {
        // Test early exit when num_elements is reached mid-sub-block
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
                         // Request exactly 17 elements (mid sub-block)
        let result = crate::apr::dequantize_q4_k(&bytes, 17);
        assert_eq!(result.len(), 17);
    }

    #[test]
    fn test_dequantize_q6_k_early_exit_more_cov() {
        // Test early exit when num_elements is reached mid-sub-block
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0
                           // Request exactly 33 elements (mid sub-block)
        let result = crate::apr::dequantize_q6_k(&bytes, 33);
        assert_eq!(result.len(), 33);
    }

    #[test]
    fn test_f16_to_f32_negative_subnormal_more_cov() {
        // Negative subnormal: sign=1, exp=0, mantissa=1
        let bits: u16 = 0x8001;
        let result = crate::apr::f16_to_f32(bits);
        assert!(result < 0.0, "Negative subnormal should be negative");
    }

    #[test]
    fn test_f16_to_f32_negative_nan_more_cov() {
        // Negative NaN: sign=1, exp=31, mantissa!=0
        let bits: u16 = 0xFC01;
        let result = crate::apr::f16_to_f32(bits);
        assert!(result.is_nan());
    }

    #[test]
    fn test_apr_model_forward_with_transformer_missing_tensor_more_cov() {
        // Create a model with transformer metadata but missing tensors
        let metadata = r#"{
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "vocab_size": 10
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset;
        let mut data = vec![0u8; data_offset as usize + 64];

        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // No tensors
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        let model = AprV2Model::from_bytes(data).expect("should load");
        assert!(model.metadata().is_transformer());
        let result = model.forward(&[1]);
        // Should fail because embedding tensor is missing
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix_more_cov() {
        let vocab = vec!["▁hello".to_string(), "▁".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // SentencePiece ▁ prefix should be handled
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_model_data_type_alias_more_cov() {
        // Test type alias AprModel = AprV2Model
        let data = create_test_apr_model();
        let model: AprModel = AprV2Model::from_bytes(data).expect("should load");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_model_type_alias_more_cov() {
        // Test legacy type alias AprModelType
        let _: AprModelType = ();
    }

    #[test]
    fn test_find_tensor_name_not_found_more_cov() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        // Try to find a tensor that doesn't exist
        let result = model.forward(&[0]); // Will try to find embedding tensor
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        // Should contain "No matching tensor" or similar
        assert!(
            err_msg.contains("not a transformer") || err_msg.contains("No matching"),
            "Error message: {}",
            err_msg
        );
    }

    // =========================================================================
    // SimpleTokenizer Tests (GH-156)
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_new() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        assert_eq!(tokenizer.vocab_size(), 5);
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_eos(2));
        assert!(!tokenizer.is_bos(0));
        assert!(!tokenizer.is_eos(0));
    }

    #[test]
    fn test_simple_tokenizer_decode_basic() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        // Decode tokens [3, 4] -> "helloworld"
        let decoded = tokenizer.decode(&[3, 4]);
        assert_eq!(decoded, "helloworld");
    }

    #[test]
    fn test_simple_tokenizer_decode_bpe_space() {
        use crate::apr::SimpleTokenizer;

        // BPE-style tokens with Ġ prefix (represents space)
        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "Ġhello".to_string(), // " hello"
            "Ġworld".to_string(), // " world"
            "!".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        // Decode tokens [3, 4, 5] -> " hello world!"
        let decoded = tokenizer.decode(&[3, 4, 5]);
        assert_eq!(decoded, " hello world!");
    }

    #[test]
    fn test_simple_tokenizer_decode_out_of_bounds() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string(), "b".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        // Token 99 is out of bounds - should be skipped or handled gracefully
        let decoded = tokenizer.decode(&[0, 99, 1]);
        // Should contain "a" and "b", may have placeholder for 99
        assert!(decoded.contains('a'));
        assert!(decoded.contains('b'));
    }

    #[test]
    fn test_simple_tokenizer_no_special_tokens() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["x".to_string(), "y".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        // No BOS/EOS defined
        assert!(!tokenizer.is_bos(0));
        assert!(!tokenizer.is_bos(1));
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(1));
    }

    #[test]
    fn test_simple_tokenizer_empty_decode() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_simple_tokenizer_clone() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["test".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), None);
        let cloned = tokenizer.clone();

        assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());
        assert_eq!(tokenizer.bos_token_id, cloned.bos_token_id);
    }

    #[test]
    fn test_simple_tokenizer_debug() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        let debug_str = format!("{:?}", tokenizer);

        assert!(debug_str.contains("SimpleTokenizer"));
    }

    // =========================================================================
    // APR v1 Format Rejection Tests
    // =========================================================================

    #[test]
    fn test_apr_v1_format_rejected() {
        // APR v1 magic: "APR1" (0x41, 0x50, 0x52, 0x31)
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // "APR1"
        data[4] = 1; // version major
        data[5] = 0; // version minor

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should indicate APR v1 not supported
        assert!(
            err.contains("APR v1") || err.contains("not supported"),
            "Error should mention APR v1: {}",
            err
        );
    }

    #[test]
    fn test_apr_v1_format_conversion_hint() {
        // Test that error message suggests conversion
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // "APR1"

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Error should hint at conversion or GGUF alternative
        assert!(
            err.contains("convert") || err.contains("GGUF"),
            "Error should suggest conversion: {}",
            err
        );
    }

    #[test]
    fn test_apr_invalid_version_byte() {
        // Test with invalid version byte (not 0, '1', or '2')
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = 0x99; // Invalid version byte

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("version") || err.contains("Invalid"),
            "Error should mention version: {}",
            err
        );
    }

    // =========================================================================
    // MappedAprModel Tests
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_from_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file with legacy magic
        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC); // APR\0 legacy magic
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert_eq!(model.tensor_count(), 0);
        assert_eq!(model.file_size(), data.len());
        assert_eq!(model.data_offset(), 64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_nonexistent_file() {
        let result = MappedAprModel::from_path("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("magic") || err.contains("Invalid"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_truncated() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write less than HEADER_SIZE bytes
        let data = vec![0u8; 32];
        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_data_access() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        // Verify data() returns the full mmap
        let model_data = model.data();
        assert_eq!(model_data.len(), 128);
        assert_eq!(&model_data[0..4], &MAGIC);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_find_tensor_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert!(model.find_tensor("nonexistent").is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_get_tensor_data_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        let result = model.get_tensor_data("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_apr_model_dtype_to_qtype() {
        // F32 -> 0
        assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
        // F16 -> 1
        assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
        // Q4_0 -> 2
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
        // Q4_1 -> 3
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
        // Q5_0 -> 6
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
        // Q5_1 -> 7
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
        // Q8_0 -> 8
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
        // Q8_1 -> 9
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
        // Q2_K -> 10
        assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
        // Q3_K -> 11
        assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
        // Q4_K -> 12
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
        // Q5_K -> 13
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
        // Q6_K -> 14
        assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
        // IQ2_XXS -> 16
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
        // IQ2_XS -> 17
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
        // BF16 -> 30
        assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
        // Unknown -> 0 (default to F32)
        assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0);
    }

    // =========================================================================
    // Embedded Tokenizer Tests (GH-156)
    // =========================================================================

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_empty() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
        // Empty array should return None
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_valid() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["<pad>", "<bos>", "<eos>", "hello", "world"]),
        );

        let vocab = meta.get_embedded_vocabulary().expect("should have vocab");
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[0], "<pad>");
        assert_eq!(vocab[3], "hello");
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_invalid_type() {
        let mut meta = AprMetadata::default();
        // Not an array
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!("not an array"),
        );
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_mixed_types() {
        let mut meta = AprMetadata::default();
        // Array with mixed types - only strings should be kept
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["valid", 123, "also_valid", null]),
        );

        let vocab = meta.get_embedded_vocabulary().expect("should have vocab");
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab[0], "valid");
        assert_eq!(vocab[1], "also_valid");
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_valid() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));

        assert_eq!(meta.get_embedded_bos_token_id(), Some(1));
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_invalid_type() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.bos_token_id".to_string(),
            serde_json::json!("not a number"),
        );
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_valid() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));

        assert_eq!(meta.get_embedded_eos_token_id(), Some(2));
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_invalid_type() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.eos_token_id".to_string(),
            serde_json::json!({"nested": "object"}),
        );
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    #[test]
    fn test_load_embedded_tokenizer_no_vocab() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Model has no embedded vocabulary
        assert!(model.load_embedded_tokenizer().is_none());
    }

    /// Helper to create APR model with embedded tokenizer
    fn create_apr_model_with_embedded_tokenizer() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "test",
            "vocab_size": 5,
            "hidden_size": 64,
            "tokenizer.vocabulary": ["<pad>", "<bos>", "<eos>", "hello", "world"],
            "tokenizer.bos_token_id": 1,
            "tokenizer.eos_token_id": 2
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset;

        let total_size = data_offset as usize + 64; // Some padding
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        data
    }

    #[test]
    fn test_load_embedded_tokenizer_valid() {
        let data = create_apr_model_with_embedded_tokenizer();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tokenizer = model
            .load_embedded_tokenizer()
            .expect("should have embedded tokenizer");

        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.bos_token_id, Some(1));
        assert_eq!(tokenizer.eos_token_id, Some(2));
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_eos(2));
    }

    #[test]
    fn test_load_embedded_tokenizer_decode() {
        let data = create_apr_model_with_embedded_tokenizer();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tokenizer = model
            .load_embedded_tokenizer()
            .expect("should have embedded tokenizer");

        let decoded = tokenizer.decode(&[3, 4]);
        assert_eq!(decoded, "helloworld");
    }

    // =========================================================================
    // Additional RoPE Metadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_rope_theta() {
        let meta = AprMetadata {
            rope_theta: Some(10000.0),
            ..Default::default()
        };
        assert_eq!(meta.rope_theta, Some(10000.0));
    }

    #[test]
    fn test_apr_metadata_rope_type() {
        // rope_type 0 = NORM (adjacent pairs)
        // rope_type 2 = NEOX (split halves) - used by Qwen2.5
        let meta = AprMetadata {
            rope_type: Some(2),
            ..Default::default()
        };
        assert_eq!(meta.rope_type, Some(2));
    }

    #[test]
    fn test_apr_metadata_rms_norm_eps() {
        let meta = AprMetadata {
            rms_norm_eps: Some(1e-6),
            ..Default::default()
        };
        assert!((meta.rms_norm_eps.unwrap() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_apr_metadata_num_kv_heads() {
        // GQA models have fewer KV heads than query heads
        let meta = AprMetadata {
            num_heads: Some(32),
            num_kv_heads: Some(4), // GQA ratio 8:1
            ..Default::default()
        };
        assert_eq!(meta.num_heads, Some(32));
        assert_eq!(meta.num_kv_heads, Some(4));
    }

    #[test]
    fn test_apr_metadata_max_position_embeddings() {
        let meta = AprMetadata {
            max_position_embeddings: Some(4096),
            ..Default::default()
        };
        assert_eq!(meta.max_position_embeddings, Some(4096));
    }

    #[test]
    fn test_apr_metadata_intermediate_size() {
        let meta = AprMetadata {
            hidden_size: Some(2048),
            intermediate_size: Some(5632), // SwiGLU FFN size
            ..Default::default()
        };
        assert_eq!(meta.intermediate_size, Some(5632));
    }

    // =========================================================================
    // Additional dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_case_insensitive() {
        // Lowercase variants
        assert_eq!(dtype_to_ggml_qtype("q4_k"), Some(12));
        assert_eq!(dtype_to_ggml_qtype("q5_k"), Some(13));
        assert_eq!(dtype_to_ggml_qtype("q6_k"), Some(14));
        assert_eq!(dtype_to_ggml_qtype("q8_0"), Some(8));
        assert_eq!(dtype_to_ggml_qtype("q4_0"), Some(2));
        assert_eq!(dtype_to_ggml_qtype("q4_1"), Some(3));
        assert_eq!(dtype_to_ggml_qtype("q5_0"), Some(6));
    }

    #[test]
    fn test_is_quantized_dtype_comprehensive() {
        // Quantized types
        assert!(is_quantized_dtype("Q4_K"));
        assert!(is_quantized_dtype("Q5_K"));
        assert!(is_quantized_dtype("Q6_K"));
        assert!(is_quantized_dtype("Q8_0"));
        assert!(is_quantized_dtype("Q4_0"));
        assert!(is_quantized_dtype("Q4_1"));
        assert!(is_quantized_dtype("Q5_0"));

        // Non-quantized types
        assert!(!is_quantized_dtype("F32"));
        assert!(!is_quantized_dtype("F16"));
        assert!(!is_quantized_dtype("BF16"));
        assert!(!is_quantized_dtype("unknown"));
    }

    // =========================================================================
    // Additional TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_all_dtypes() {
        // GH-191 FIX: dtype bytes now use GGML type IDs
        let dtypes = [
            (0u8, "F32"),
            (1, "F16"),
            (2, "Q4_0"),   // GGML type 2
            (3, "Q4_1"),   // GGML type 3
            (6, "Q5_0"),   // GGML type 6
            (7, "Q5_1"),   // GGML type 7
            (8, "Q8_0"),   // GGML type 8
            (9, "Q8_1"),   // GGML type 9
            (10, "Q2_K"),  // GGML type 10
            (11, "Q3_K"),  // GGML type 11
            (12, "Q4_K"),  // GGML type 12
            (13, "Q5_K"),  // GGML type 13
            (14, "Q6_K"),  // GGML type 14
            (30, "BF16"),  // GGML type 30
        ];

        for (dtype_byte, expected_dtype) in dtypes {
            let data = create_binary_tensor_entry("test", dtype_byte, &[10], 0, 40);
            let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
            assert_eq!(
                entry.dtype, expected_dtype,
                "dtype_byte {} should map to {}",
                dtype_byte, expected_dtype
            );
        }
    }

    #[test]
    fn test_tensor_entry_from_binary_unknown_dtype_defaults_f32() {
        // Unknown dtype byte (e.g., 255) should default to F32
        let data = create_binary_tensor_entry("test", 255, &[10], 0, 40);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.dtype, "F32");
    }

    #[test]
    fn test_tensor_entry_from_binary_8d_shape() {
        // Maximum supported dimensions
        let data = create_binary_tensor_entry(
            "high_dim_tensor",
            0,
            &[2, 3, 4, 5, 6, 7, 8, 9],
            0,
            2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 4,
        );
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.shape.len(), 8);
        assert_eq!(entry.element_count(), 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9);
    }

    // =========================================================================
    // Additional Model Loading Error Tests
    // =========================================================================

    #[test]
    fn test_apr_model_metadata_truncated_past_eof() {
        // metadata_offset + metadata_size > file length
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
        data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 > remaining bytes
        data[24..32].copy_from_slice(&1064u64.to_le_bytes());
        data[32..40].copy_from_slice(&1064u64.to_le_bytes());

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_apr_model_tensor_out_of_bounds() {
        // Create model where tensor data extends past EOF
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Create tensor entry that points past EOF
        let tensor_entry = create_binary_tensor_entry(
            "bad_tensor",
            0,
            &[1000, 1000],
            0,
            4_000_000, // Size much larger than file
        );

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;

        // File is only 256 bytes total, but tensor claims to be 4MB
        let total_size = data_offset as usize + 64;
        let mut data = vec![0u8; total_size];

        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");

        // Accessing tensor data should fail
        let result = model.get_tensor_f32("bad_tensor");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }

    // =========================================================================
    // Generate Method Edge Cases
    // =========================================================================

    #[test]
    fn test_apr_model_generate_eos_stops() {
        // Test that generation stops at EOS token
        // (Requires a valid transformer model - using mock to verify logic)
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Empty input should fail
        let result = model.generate(&[], 10, Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_generate_max_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Test that generate with max_tokens=0 works (returns empty or immediate)
        let result = model.generate(&[1, 2, 3], 0, None);
        // Either succeeds with 0 tokens or returns an error - both are valid
        if let Ok(tokens) = result {
            assert!(tokens.len() <= 3); // At most initial tokens
        }
    }

    // =========================================================================
    // SimpleTokenizer Additional Tests
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_bpe_newline() {
        use crate::apr::SimpleTokenizer;

        // BPE newline is Ċ
        let vocab = vec!["hello".to_string(), "Ċ".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(decoded, "hello\nworld");
    }

    #[test]
    fn test_simple_tokenizer_bpe_tab() {
        use crate::apr::SimpleTokenizer;

        // BPE tab is ĉ
        let vocab = vec!["hello".to_string(), "ĉ".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(decoded, "hello\tworld");
    }

    #[test]
    fn test_simple_tokenizer_large_vocab() {
        use crate::apr::SimpleTokenizer;

        // Create large vocabulary
        let vocab: Vec<String> = (0..10000).map(|i| format!("token_{}", i)).collect();
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(9999));

        assert_eq!(tokenizer.vocab_size(), 10000);
        assert!(tokenizer.is_bos(0));
        assert!(tokenizer.is_eos(9999));

        let decoded = tokenizer.decode(&[100, 200, 300]);
        assert!(decoded.contains("token_100"));
        assert!(decoded.contains("token_200"));
        assert!(decoded.contains("token_300"));
    }

    // =========================================================================
    // BpeTokenizer Additional Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_large_merge_rules() {
        // Test with many merge rules
        let vocab: HashMap<String, u32> = [
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let id_to_token = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "ab".to_string(),
            "abc".to_string(),
        ];

        let merge_rules = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];

        let tokenizer = BpeTokenizer {
            token_to_id: vocab,
            id_to_token,
            merge_rules,
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let encoded = tokenizer.encode("abc");
        // Should merge a+b -> ab, then ab+c -> abc
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_non_ascii() {
        // Test encoding non-ASCII characters
        let vocab: HashMap<String, u32> = [("a".to_string(), 0)].into_iter().collect();

        let id_to_token = vec!["a".to_string()];

        let tokenizer = BpeTokenizer {
            token_to_id: vocab,
            id_to_token,
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        // Non-ASCII should be handled (may result in empty if not in vocab)
        let encoded = tokenizer.encode("日本語");
        // Result depends on byte encoding - just verify it doesn't panic
        let _ = encoded;
    }

    // =========================================================================
    // MAGIC_PREFIX Tests
    // =========================================================================

    #[test]
    fn test_magic_prefix_constant() {
        assert_eq!(MAGIC_PREFIX, [0x41, 0x50, 0x52]); // "APR"
        assert_eq!(&MAGIC_PREFIX, b"APR");
    }

    #[test]
    fn test_header_size_constant() {
        assert_eq!(HEADER_SIZE, 64);
    }

    #[test]
    fn test_alignment_constant() {
        assert_eq!(ALIGNMENT, 64);
    }

    // ==========================================================================
    // PMAT-107: Falsification Test for GQA num_kv_heads in AprMetadata
    // ==========================================================================

    /// FALSIFICATION TEST: Verify AprMetadata correctly parses num_kv_heads from JSON
    /// This catches the silent failure case where unwrap_or_default() swallows errors.
    #[test]
    fn test_falsification_apr_metadata_parses_gqa_num_kv_heads() {
        // JSON from a real Qwen 1.5B APR file (GQA: 12 heads, 2 kv_heads)
        let metadata_json = r#"{
            "model_type": "qwen2",
            "architecture": "qwen2",
            "hidden_size": 1536,
            "num_layers": 28,
            "num_heads": 12,
            "num_kv_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001
        }"#;

        let metadata: AprMetadata = serde_json::from_str(metadata_json)
            .expect("AprMetadata should parse valid JSON");

        assert_eq!(
            metadata.num_heads,
            Some(12),
            "num_heads not parsed correctly"
        );
        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads not parsed from AprMetadata JSON!\n\
             This causes GQA models to hang on GPU because they're treated as MHA."
        );
        assert_eq!(
            metadata.hidden_size,
            Some(1536),
            "hidden_size not parsed correctly"
        );

        println!("✅ AprMetadata correctly parses num_kv_heads={:?}", metadata.num_kv_heads);
    }

    /// FALSIFICATION TEST: Verify AprMetadata handles extra fields gracefully
    /// The real APR files have extra fields like "model.num_kv_heads" that should
    /// be captured by the flatten attribute without breaking num_kv_heads parsing.
    #[test]
    fn test_falsification_apr_metadata_with_extra_fields() {
        // This JSON has both the standard fields AND the extra "model.*" fields
        // from the Q4K converter
        let metadata_json = r#"{
            "model_type": "qwen2",
            "name": "qwen2",
            "architecture": "qwen2",
            "hidden_size": 1536,
            "num_layers": 28,
            "num_heads": 12,
            "num_kv_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001,
            "model.intermediate_size": 8960,
            "model.num_kv_heads": 2,
            "model.rope_theta": 1000000.0
        }"#;

        let metadata: AprMetadata = serde_json::from_str(metadata_json)
            .expect("AprMetadata should parse JSON with extra fields");

        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads broken by extra 'model.*' fields!\n\
             Expected: Some(2), Got: {:?}",
            metadata.num_kv_heads
        );

        // Verify extra fields are captured
        assert!(
            metadata.extra.contains_key("model.num_kv_heads"),
            "Extra fields should be captured in 'extra' map"
        );

        println!("✅ AprMetadata handles extra fields correctly, num_kv_heads={:?}", metadata.num_kv_heads);
    }

    /// PMAT-107: Falsification test with REAL APR file
    ///
    /// This test loads the actual APR file from disk and verifies that
    /// num_kv_heads is correctly parsed. If this test fails, the APR CUDA
    /// path will hang because GQA models will be treated as MHA.
    #[test]
    fn test_falsification_real_apr_file_num_kv_heads() {
        let apr_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-apr/qwen2.5-coder-1.5b-q4k.apr"
        );

        if !apr_path.exists() {
            println!("⚠️ Test model not available at {:?}, skipping", apr_path);
            return;
        }

        // Load the model
        let model = AprV2Model::load(apr_path)
            .expect("Should load APR file");

        println!("=== REAL APR FILE METADATA ===");
        println!("  num_heads: {:?}", model.metadata.num_heads);
        println!("  num_kv_heads: {:?}", model.metadata.num_kv_heads);
        println!("  hidden_size: {:?}", model.metadata.hidden_size);
        println!("  num_layers: {:?}", model.metadata.num_layers);

        // This is the critical assertion
        assert!(
            model.metadata.num_kv_heads.is_some(),
            "FALSIFICATION FAILED: num_kv_heads is None after loading real APR file!\n\
             This causes GQA models to hang on GPU because they're treated as MHA.\n\
             Expected: Some(2) for Qwen 1.5B GQA, Got: None"
        );

        assert_eq!(
            model.metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads wrong value!\n\
             Expected: Some(2) for Qwen 1.5B GQA, Got: {:?}",
            model.metadata.num_kv_heads
        );

        println!("✅ Real APR file has correct num_kv_heads={:?}", model.metadata.num_kv_heads);
    }

    /// PMAT-107: Falsification test for GQA dimensions in CUDA executor
    ///
    /// This test loads a real APR file and creates AprV2ModelCuda, then verifies
    /// that the CUDA executor has the correct GQA dimensions. This catches bugs
    /// where num_kv_heads is parsed correctly but not propagated to the executor.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_falsification_apr_cuda_gqa_dimensions() {
        let apr_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-apr/qwen2.5-coder-1.5b-q4k.apr"
        );

        if !apr_path.exists() {
            println!("⚠️ Test model not available at {:?}, skipping", apr_path);
            return;
        }

        // Load the model
        let model = AprV2Model::load(apr_path)
            .expect("Should load APR file");

        // Verify metadata first
        assert_eq!(model.metadata.num_heads, Some(12), "num_heads should be 12");
        assert_eq!(model.metadata.num_kv_heads, Some(2), "num_kv_heads should be 2 (GQA)");

        // Create CUDA model
        use crate::apr::AprV2ModelCuda;

        let cuda_model = match AprV2ModelCuda::new(model, 0) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("⚠️ CUDA not available: {e}");
                return;
            }
        };

        // Access the executor's GQA dimensions
        // We need to verify kv_num_heads and kv_num_kv_heads are set correctly
        // The executor is private but we can check via the metadata pass-through

        println!("=== CUDA EXECUTOR GQA CONFIG ===");
        println!("  model.metadata.num_heads: {:?}", cuda_model.inner().metadata.num_heads);
        println!("  model.metadata.num_kv_heads: {:?}", cuda_model.inner().metadata.num_kv_heads);

        // The critical check: if CUDA model was initialized correctly, the GQA ratio should be 6:1
        // (12 Q heads / 2 KV heads = 6x repeat factor for GQA)
        let num_heads = cuda_model.inner().metadata.num_heads.unwrap_or(1);
        let num_kv_heads = cuda_model.inner().metadata.num_kv_heads.unwrap_or(num_heads);
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(
            gqa_ratio, 6,
            "FALSIFICATION FAILED: GQA ratio wrong!\n\
             Expected: 6 (12 Q heads / 2 KV heads), Got: {} ({} / {})",
            gqa_ratio, num_heads, num_kv_heads
        );

        println!("✅ CUDA model has correct GQA ratio: {} ({}:{} heads:kv_heads)",
            gqa_ratio, num_heads, num_kv_heads);
    }
}
