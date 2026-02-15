
impl AprTransformer {

    /// Generate tokens using KV cache (delegates to generation module)
    pub fn generate_with_cache(&self, prompt: &[u32], config: &GenerateConfig) -> Result<Vec<u32>> {
        generation::generate_with_cache(self, prompt, config)
    }
}

include!("mod_part_02_part_02.rs");
include!("mod_part_02_part_03.rs");
include!("mod_part_02_part_04.rs");
include!("mod_part_02_part_05.rs");
include!("mod_part_02_part_06.rs");
include!("mod_part_02_part_07.rs");
