//! PMAT-044: CUDA batch scheduler for continuous batching on `/v1/chat/completions`
//!
//! Accumulates streaming requests and processes them in batches using
//! `generate_batched_streaming` for weight sharing across concurrent requests.

#[cfg(feature = "cuda")]
use crate::gguf::{OwnedQuantizedModelCuda, QuantizedGenerateConfig};
use std::sync::Arc;

/// Request submitted to the batch scheduler
#[cfg(feature = "cuda")]
pub struct CudaBatchRequest {
    /// Tokenized prompt IDs
    pub prompt_ids: Vec<u32>,
    /// Generation configuration (max tokens, temperature, stop tokens)
    pub config: QuantizedGenerateConfig,
    /// Channel to stream generated token IDs back to the HTTP handler
    pub token_tx: tokio::sync::mpsc::Sender<Result<u32, String>>,
}

/// PMAT-044: Batch scheduler configuration
#[cfg(feature = "cuda")]
pub struct CudaBatchConfig {
    /// Maximum batch size (default 4)
    pub max_batch: usize,
    /// Window timeout in ms — how long to wait for batch to fill (default 10ms)
    pub window_ms: u64,
}

#[cfg(feature = "cuda")]
impl Default for CudaBatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 4,
            window_ms: 10,
        }
    }
}

/// Spawn the batch scheduler background task.
///
/// Returns a sender for submitting requests.
#[cfg(feature = "cuda")]
pub fn spawn_cuda_batch_scheduler(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: CudaBatchConfig,
) -> tokio::sync::mpsc::Sender<CudaBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel::<CudaBatchRequest>(256);

    // Run the scheduler in a blocking thread (CUDA ops are synchronous)
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("PMAT-044: failed to create scheduler runtime");

        rt.block_on(async move {
            cuda_batch_scheduler_loop(model, config, rx).await;
        });
    });

    tx
}

#[cfg(feature = "cuda")]
async fn cuda_batch_scheduler_loop(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: CudaBatchConfig,
    mut rx: tokio::sync::mpsc::Receiver<CudaBatchRequest>,
) {
    eprintln!(
        "[PMAT-044] Batch scheduler started: max_batch={}, window={}ms",
        config.max_batch, config.window_ms
    );

    loop {
        // Wait for at least one request
        let first = match rx.recv().await {
            Some(req) => req,
            None => {
                eprintln!("[PMAT-044] Batch scheduler shutting down (channel closed)");
                return;
            },
        };

        // Accumulate more requests within the window
        let mut batch = vec![first];
        let deadline =
            tokio::time::Instant::now() + tokio::time::Duration::from_millis(config.window_ms);

        while batch.len() < config.max_batch {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                Ok(None) => {
                    eprintln!("[PMAT-044] Channel closed during accumulation");
                    // Process remaining batch then exit
                    if !batch.is_empty() {
                        process_cuda_batch(&model, batch);
                    }
                    return;
                },
                Err(_timeout) => break, // Window expired
            }
        }

        let batch_size = batch.len();
        eprintln!("[PMAT-044] Processing batch of {} requests", batch_size);

        // Process the batch
        process_cuda_batch(&model, batch);
    }
}

#[cfg(feature = "cuda")]
fn process_cuda_batch(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    batch: Vec<CudaBatchRequest>,
) {
    let m = batch.len();

    if m == 1 {
        // Single request — use the optimized single-request path
        let req = batch.into_iter().next().unwrap();
        let mut cuda_model = model.write().expect("PMAT-044: model lock poisoned");
        let result =
            cuda_model.generate_gpu_resident_streaming(&req.prompt_ids, &req.config, |token_id| {
                req.token_tx.try_send(Ok(token_id)).is_ok()
            });
        if let Err(e) = result {
            let _ = req.token_tx.try_send(Err(e.to_string()));
        }
        return;
    }

    // Multi-request — use batched streaming
    let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_ids.clone()).collect();
    let configs: Vec<QuantizedGenerateConfig> = batch.iter().map(|r| r.config.clone()).collect();

    // Create per-slot streaming callbacks
    let senders: Vec<tokio::sync::mpsc::Sender<Result<u32, String>>> =
        batch.iter().map(|r| r.token_tx.clone()).collect();

    let callbacks: Vec<Box<dyn FnMut(u32) -> bool + Send>> = senders
        .into_iter()
        .map(|tx| {
            Box::new(move |token_id: u32| -> bool { tx.try_send(Ok(token_id)).is_ok() })
                as Box<dyn FnMut(u32) -> bool + Send>
        })
        .collect();

    let mut cuda_model = model.write().expect("PMAT-044: model lock poisoned");
    let result = cuda_model.generate_batched_streaming(&prompts, &configs, callbacks);

    if let Err(e) = result {
        // Send error to all slots
        for req in &batch {
            let _ = req.token_tx.try_send(Err(e.to_string()));
        }
    }
}
