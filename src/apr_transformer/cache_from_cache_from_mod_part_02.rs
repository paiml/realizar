impl AprTransformer {
    /// Generate tokens using KV cache (delegates to generation module)
    pub fn generate_with_cache(&self, prompt: &[u32], config: &GenerateConfig) -> Result<Vec<u32>> {
        generation::generate_with_cache(self, prompt, config)
    }

    /// Generate tokens with streaming callback (GH-284)
    ///
    /// Same as `generate_with_cache` but calls `on_token` after each token.
    /// Return `false` from the callback to stop early (client disconnected).
    pub fn generate_with_cache_streaming<F>(
        &self,
        prompt: &[u32],
        config: &GenerateConfig,
        on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        generation::generate_with_cache_streaming(self, prompt, config, on_token)
    }
}

include!("from_apr_file.rs");
include!("mod_part_02_part_03.rs");
include!("mod_part_02_part_04.rs");
include!("pmat-260.rs");
include!("mod_part_02_part_06.rs");
include!("cache_from_mod_part_02_part_07.rs");
