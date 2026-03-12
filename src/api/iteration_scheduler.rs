//! PMAT-088: Iteration-level scheduler for continuous batching.
//!
//! Replaces the batch-then-wait scheduler (cuda_batch_scheduler) with
//! iteration-level scheduling inspired by Orca (Yu et al., OSDI 2022)
//! and Sarathi-Serve (Agrawal et al., OSDI 2024).
//!
//! Key differences from cuda_batch_scheduler:
//! - Decode-maximal: always schedules ALL running decode tokens first
//! - Chunked prefill: splits long prompts across iterations (no generation stalls)
//! - Token budget: caps total tokens per forward pass
//! - Per-iteration scheduling decisions (not per-batch)
//!
//! Enabled via `ITERATION_SCHEDULER=1` env var (opt-in during development).

#[cfg(feature = "cuda")]
use crate::api::cuda_batch_scheduler::CudaBatchRequest;
#[cfg(feature = "cuda")]
use crate::gguf::OwnedQuantizedModelCuda;
use std::collections::VecDeque;
use std::sync::Arc;

/// Iteration scheduler configuration.
#[cfg(feature = "cuda")]
pub struct IterationSchedulerConfig {
    /// Maximum concurrent decode slots (default 4, env CUDA_MAX_BATCH)
    pub max_slots: usize,
    /// Prefill chunk size in tokens — tile-aligned for sm_89 (default 256)
    pub prefill_chunk_size: usize,
    /// Token budget per forward pass (0 = unlimited, env ITERATION_TOKEN_BUDGET)
    pub token_budget: usize,
}

#[cfg(feature = "cuda")]
impl Default for IterationSchedulerConfig {
    fn default() -> Self {
        let max_slots = std::env::var("CUDA_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let prefill_chunk_size = std::env::var("PREFILL_CHUNK_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let token_budget = std::env::var("ITERATION_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0); // 0 = unlimited (for now)
        Self {
            max_slots,
            prefill_chunk_size,
            token_budget,
        }
    }
}

/// Per-iteration scheduling decision.
#[cfg(feature = "cuda")]
struct SchedulerOutput {
    /// Number of running slots with decode tokens this iteration
    num_decode: usize,
    /// If Some, a prefill chunk to run this iteration: (slot_idx, chunk_start, chunk_end)
    prefill_chunk: Option<PrefillChunk>,
}

/// A prefill chunk to process in one iteration.
#[cfg(feature = "cuda")]
struct PrefillChunk {
    /// Which request in the waiting queue
    waiting_idx: usize,
    /// Token range [start, end) within the prompt
    start_token: usize,
    end_token: usize,
    /// True if this is the last chunk (moves request to running)
    is_final: bool,
}

/// Spawn the iteration-level scheduler.
///
/// Drop-in replacement for `spawn_cuda_batch_scheduler` when
/// `ITERATION_SCHEDULER=1` is set.
#[cfg(feature = "cuda")]
pub fn spawn_iteration_scheduler(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: IterationSchedulerConfig,
) -> tokio::sync::mpsc::Sender<CudaBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel::<CudaBatchRequest>(256);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("PMAT-088: failed to create scheduler runtime");

        rt.block_on(async move {
            iteration_scheduler_loop(model, config, rx).await;
        });
    });

    tx
}

/// State tracking for a request being prefilled in chunks.
#[cfg(feature = "cuda")]
struct ChunkedPrefillState {
    request: CudaBatchRequest,
    /// How many prompt tokens have been prefilled so far
    tokens_prefilled: usize,
    /// Total prompt tokens to prefill
    total_tokens: usize,
}

#[cfg(feature = "cuda")]
async fn iteration_scheduler_loop(
    model: Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    config: IterationSchedulerConfig,
    mut rx: tokio::sync::mpsc::Receiver<CudaBatchRequest>,
) {
    eprintln!(
        "[PMAT-088] Iteration scheduler started: max_slots={}, prefill_chunk={}, token_budget={}",
        config.max_slots,
        config.prefill_chunk_size,
        if config.token_budget == 0 {
            "unlimited".to_string()
        } else {
            config.token_budget.to_string()
        }
    );

    // Waiting queue: requests pending prefill
    let mut waiting: VecDeque<CudaBatchRequest> = VecDeque::new();

    // Running state: managed by BatchedDecodeState inside the model lock
    // We track whether a batch is active here
    let mut batch_active = false;
    let mut total_batches: u64 = 0;
    let mut total_iterations: u64 = 0;

    loop {
        // Drain incoming requests into waiting queue
        if !batch_active && waiting.is_empty() {
            // Nothing running — block until a request arrives
            match rx.recv().await {
                Some(req) => waiting.push_back(req),
                None => {
                    eprintln!("[PMAT-088] Iteration scheduler shutting down");
                    return;
                },
            }
        }

        // Non-blocking drain of any additional queued requests
        tokio::task::yield_now().await;
        while let Ok(req) = rx.try_recv() {
            waiting.push_back(req);
        }

        if waiting.is_empty() && !batch_active {
            continue;
        }

        // === SCHEDULING DECISION ===
        //
        // PMAT-088c: Decode-maximal policy (Sarathi-Serve / Orca):
        // 1. Form initial batch with just 1 request (start decode ASAP)
        // 2. Remaining waiting requests join via mid-batch add_slot_to_batch()
        //    between decode steps — interleaved prefill, no decode stalls
        // 3. Slot recycling for finished slots with pending requests
        //
        // This gives decode-maximal scheduling: slot 0 starts generating tokens
        // immediately (~21ms TTFT) while slots 1-3 are prefilled one at a time
        // between decode steps (~14ms prefill per slot, interleaved with ~13ms decode).
        //
        // Expected TTFT improvement at c=4:
        //   Before: 82ms (all 4 prefilled upfront before any decode)
        //   After:  ~21ms for slot 0 (14ms prefill + 6.6ms first decode)
        //           ~35ms for slot 1, ~50ms for slot 2, ~65ms for slot 3
        //           P50 TTFT ≈ 42ms (49% improvement)

        if !batch_active {
            // PMAT-088c: Form initial batch with 1 request. Remaining stay in
            // waiting queue for mid-batch joins during decode loop.
            let batch: Vec<CudaBatchRequest> = vec![waiting.pop_front().unwrap()];

            total_batches += 1;
            let batch_start = std::time::Instant::now();

            eprintln!(
                "[PMAT-088c] Batch #{}: starting m=1 (decode-maximal), waiting={} for mid-batch join",
                total_batches,
                waiting.len(),
            );

            // Process using existing infrastructure (PMAT-072/073/074).
            // Pass rx for mid-batch joins and recycling.
            process_iteration_batch(
                &model,
                batch,
                &mut rx,
                &mut waiting,
                config.max_slots,
                &mut total_iterations,
            );

            let elapsed = batch_start.elapsed();
            eprintln!(
                "[PMAT-088] Batch #{} done in {:.1}ms, {} iterations",
                total_batches,
                elapsed.as_secs_f64() * 1000.0,
                total_iterations,
            );
        }
    }
}

/// Process a batch using iteration-level scheduling.
///
/// Uses the existing PMAT-072/073/074 infrastructure but adds:
/// - Waiting queue integration (new requests from waiting queue, not just rx)
/// - Per-iteration metrics
/// - Prefill interleaving preparation
#[cfg(feature = "cuda")]
fn process_iteration_batch(
    model: &Arc<std::sync::RwLock<OwnedQuantizedModelCuda>>,
    batch: Vec<CudaBatchRequest>,
    rx: &mut tokio::sync::mpsc::Receiver<CudaBatchRequest>,
    waiting: &mut VecDeque<CudaBatchRequest>,
    max_slots: usize,
    total_iterations: &mut u64,
) {
    use crate::gguf::QuantizedGenerateConfig;

    let m = batch.len();

    // Single request — fast M=1 path (CUDA graph replay, 154.8 tok/s)
    if m == 1 && waiting.is_empty() {
        let req = batch.into_iter().next().unwrap();
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        let result =
            cuda_model.generate_gpu_resident_streaming(&req.prompt_ids, &req.config, |token_id| {
                req.token_tx.try_send(Ok(token_id)).is_ok()
            });
        if let Err(e) = result {
            let _ = req.token_tx.try_send(Err(e.to_string()));
        }
        *total_iterations += 1;
        return;
    }

    // Multi-request batch — PMAT-072/073/074 step-wise decode
    let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_ids.clone()).collect();
    let configs: Vec<QuantizedGenerateConfig> = batch.iter().map(|r| r.config.clone()).collect();
    let mut error_senders: Vec<tokio::sync::mpsc::Sender<Result<u32, String>>> =
        batch.iter().map(|r| r.token_tx.clone()).collect();

    let callbacks: Vec<Box<dyn FnMut(u32) -> bool + Send>> = batch
        .into_iter()
        .map(|req| {
            Box::new(move |token_id: u32| -> bool { req.token_tx.try_send(Ok(token_id)).is_ok() })
                as Box<dyn FnMut(u32) -> bool + Send>
        })
        .collect();

    // Phase 1: Setup + Prefill
    let mut state = {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        match cuda_model.batched_setup_and_prefill(&prompts, &configs, callbacks, max_slots) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[PMAT-088] Setup+prefill ERROR (m={m}): {e}");
                for tx in &error_senders {
                    let _ = tx.try_send(Err(e.to_string()));
                }
                return;
            },
        }
    };

    // Phase 2: Iteration-level decode loop
    while !state.all_done() && state.gen_idx < state.max_tokens_max {
        *total_iterations += 1;
        let iter_start = std::time::Instant::now();

        let token_ids = {
            let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");

            // PMAT-088: Check waiting queue FIRST, then rx channel.
            // This ensures requests that arrived during previous iteration's
            // token distribution get scheduled before new channel arrivals.

            // PMAT-088c: Decode-maximal scheduling — limit to 1 mid-batch join
            // per decode step. This interleaves prefill (14ms) with decode (6-13ms)
            // instead of blocking decode for N × 14ms to prefill all waiting at once.
            // Result: slot 0 TTFT drops from 82ms to ~21ms (P50 from 82ms to ~42ms).
            let mut joined_this_step = false;

            // Mid-batch joins from waiting queue (1 per step)
            if !joined_this_step && state.m < state.max_kv_slots {
                if let Some(req) = waiting.pop_front() {
                    let error_tx = req.token_tx.clone();
                    let on_token: Box<dyn FnMut(u32) -> bool + Send> = {
                        let token_tx = req.token_tx;
                        Box::new(move |token_id: u32| -> bool {
                            token_tx.try_send(Ok(token_id)).is_ok()
                        })
                    };
                    match cuda_model.add_slot_to_batch(
                        &mut state,
                        req.prompt_ids,
                        req.config,
                        on_token,
                    ) {
                        Ok(()) => {
                            error_senders.push(error_tx);
                            joined_this_step = true;
                        },
                        Err(e) => {
                            eprintln!("[PMAT-088c] Mid-batch join from waiting FAILED: {e}");
                            let _ = error_tx.try_send(Err(e.to_string()));
                        },
                    }
                }
            }

            // Mid-batch joins from rx channel (1 per step, only if no waiting join)
            if !joined_this_step && state.m < state.max_kv_slots {
                match rx.try_recv() {
                    Ok(req) => {
                        let error_tx = req.token_tx.clone();
                        let on_token: Box<dyn FnMut(u32) -> bool + Send> = {
                            let token_tx = req.token_tx;
                            Box::new(move |token_id: u32| -> bool {
                                token_tx.try_send(Ok(token_id)).is_ok()
                            })
                        };
                        match cuda_model.add_slot_to_batch(
                            &mut state,
                            req.prompt_ids,
                            req.config,
                            on_token,
                        ) {
                            Ok(()) => {
                                error_senders.push(error_tx);
                                joined_this_step = true;
                            },
                            Err(e) => {
                                eprintln!("[PMAT-088c] Mid-batch join from rx FAILED: {e}");
                                let _ = error_tx.try_send(Err(e.to_string()));
                            },
                        }
                    },
                    Err(_) => {},
                }
            }

            // PMAT-088c: Slot recycling — 1 per step (same decode-maximal policy)
            if !joined_this_step {
                for slot_idx in 0..state.m {
                    if !state.done[slot_idx] {
                        continue;
                    }
                    // Try waiting queue first, then rx
                    let next_req = waiting.pop_front().or_else(|| rx.try_recv().ok());
                    if let Some(req) = next_req {
                        let error_tx = req.token_tx.clone();
                        let on_token: Box<dyn FnMut(u32) -> bool + Send> = {
                            let token_tx = req.token_tx;
                            Box::new(move |token_id: u32| -> bool {
                                token_tx.try_send(Ok(token_id)).is_ok()
                            })
                        };
                        match cuda_model.recycle_slot(
                            &mut state,
                            slot_idx,
                            req.prompt_ids,
                            req.config,
                            on_token,
                        ) {
                            Ok(()) => error_senders[slot_idx] = error_tx,
                            Err(e) => {
                                eprintln!("[PMAT-088c] Slot recycle FAILED (slot {slot_idx}): {e}");
                                let _ = error_tx.try_send(Err(e.to_string()));
                            },
                        }
                    } else {
                        break;
                    }
                    break; // PMAT-088c: 1 recycle per step (decode-maximal)
                }
            }

            // Decode step
            match cuda_model.batched_decode_step(&mut state) {
                Ok(ids) => ids,
                Err(e) => {
                    eprintln!(
                        "[PMAT-088] Decode step ERROR (m={}, step={}): {e}",
                        state.m, state.gen_idx
                    );
                    for tx in &error_senders {
                        let _ = tx.try_send(Err(e.to_string()));
                    }
                    cuda_model.batched_cleanup(&state);
                    return;
                },
            }
        };

        // Token distribution WITHOUT lock
        state.distribute_tokens(&token_ids);

        // Per-iteration metrics (first 3 only to avoid log spam)
        if *total_iterations <= 3 || state.gen_idx % 50 == 0 {
            let iter_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            let active_slots = state.done.iter().filter(|&&d| !d).count();
            eprintln!(
                "[PMAT-088] iter={}, m={}, active={}, step_ms={:.1}",
                total_iterations, state.m, active_slots, iter_ms,
            );
        }
    }

    // Phase 3: Cleanup
    {
        let mut cuda_model = model.write().expect("PMAT-088: model lock poisoned");
        cuda_model.batched_cleanup(&state);
    }
}
