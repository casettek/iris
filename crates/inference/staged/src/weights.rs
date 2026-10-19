//! A run owns its immutable weights; stage runtime state never lives here.
use crate::cache::MaterializationCache;
use crate::routines::RoutineRunCaches;
use anyhow::{Context, Result};
use inference_kernels::prefill_range::PrefillRangeWeightCache;
use inference_kernels::weight_budget::{WeightBudget, WeightCacheStats};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct RunWeights {
    budget: Arc<WeightBudget>,
    materializations: MaterializationCache,
    prepared: PrefillRangeWeightCache,
}

impl RunWeights {
    pub fn new(bytes: usize) -> Self {
        let budget = Arc::new(WeightBudget::new(bytes));
        Self {
            materializations: MaterializationCache::with_budget(Arc::clone(&budget)),
            prepared: PrefillRangeWeightCache::with_budget(Arc::clone(&budget)),
            budget,
        }
    }
    pub fn caches(&self) -> RoutineRunCaches<'_> {
        RoutineRunCaches {
            materializations: Some(&self.materializations),
            prefill_range_weights: Some(&self.prepared),
        }
    }
    pub fn stats(&self) -> WeightCacheStats {
        self.budget.stats()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightAllocation {
    pub total_bytes: usize,
    pub main_bytes: usize,
    pub worker_bytes: usize,
    pub worker_count: usize,
    pub source: String,
}

impl WeightAllocation {
    pub fn resolve(workers: usize) -> Result<Self> {
        let (total, source) = match std::env::var("STAGED_INFER_WEIGHT_CACHE_BYTES") {
            Ok(value) => (
                value
                    .parse::<usize>()
                    .context("STAGED_INFER_WEIGHT_CACHE_BYTES must be a nonnegative byte count")?,
                "environment",
            ),
            Err(std::env::VarError::NotPresent) => {
                let mut system = sysinfo::System::new();
                system.refresh_memory();
                match usize::try_from(system.available_memory())
                    .ok()
                    .filter(|n| *n > 0)
                {
                    Some(bytes) => (bytes / 4, "25% of available RAM"),
                    None => (1 << 30, "1 GiB fallback: available RAM unavailable"),
                }
            }
            Err(error) => return Err(error).context("invalid STAGED_INFER_WEIGHT_CACHE_BYTES"),
        };
        Ok(Self::split(total, workers, source))
    }
    pub fn split(total: usize, workers: usize, source: &str) -> Self {
        let main = if workers == 0 { total } else { total / 2 };
        Self {
            total_bytes: total,
            main_bytes: main,
            worker_bytes: if workers == 0 {
                0
            } else {
                (total - main) / workers
            },
            worker_count: workers,
            source: source.into(),
        }
    }
}

/// Sample combined RSS without retaining any model/runtime objects. This reports
/// a sampled peak, not an allocator-exact bound or an OS high-water mark.
pub(crate) struct MemorySampler {
    state: Arc<(
        std::sync::Mutex<(bool, Vec<sysinfo::Pid>)>,
        std::sync::Condvar,
    )>,
    peak: Arc<std::sync::atomic::AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MemorySampler {
    pub fn start() -> Self {
        let state = Arc::new((
            std::sync::Mutex::new((false, vec![sysinfo::Pid::from_u32(std::process::id())])),
            std::sync::Condvar::new(),
        ));
        let peak = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_state = Arc::clone(&state);
        let worker_peak = Arc::clone(&peak);
        let thread = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            loop {
                let guard = worker_state.0.lock().unwrap();
                if guard.0 {
                    break;
                }
                let mut pids = guard.1.clone();
                drop(guard);
                system.refresh_processes_specifics(
                    sysinfo::ProcessesToUpdate::All,
                    true,
                    sysinfo::ProcessRefreshKind::nothing().with_memory(),
                );
                // Include short-lived diagnostic subprocesses as well as the persistent pool.
                loop {
                    let children: Vec<_> = system
                        .processes()
                        .iter()
                        .filter(|(pid, process)| {
                            process.thread_kind().is_none()
                                && !pids.contains(pid)
                                && process
                                    .parent()
                                    .is_some_and(|parent| pids.contains(&parent))
                        })
                        .map(|(pid, _)| *pid)
                        .collect();
                    if children.is_empty() {
                        break;
                    }
                    pids.extend(children);
                }
                let rss = pids
                    .iter()
                    .filter_map(|pid| system.process(*pid))
                    .map(|p| p.memory())
                    .sum();
                worker_peak.fetch_max(rss, std::sync::atomic::Ordering::Relaxed);
                let guard = worker_state.0.lock().unwrap();
                if guard.0 {
                    break;
                }
                let _ = worker_state
                    .1
                    .wait_timeout(guard, std::time::Duration::from_millis(100))
                    .unwrap();
            }
        });
        Self {
            state,
            peak,
            thread: Some(thread),
        }
    }
    pub fn add_workers(&self, ids: impl IntoIterator<Item = u32>) {
        self.state
            .0
            .lock()
            .unwrap()
            .1
            .extend(ids.into_iter().map(sysinfo::Pid::from_u32));
        self.state.1.notify_one();
    }
    pub fn peak(&self) -> u64 {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for MemorySampler {
    fn drop(&mut self) {
        self.state.0.lock().unwrap().0 = true;
        self.state.1.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_never_exceeds_total() {
        for total in [0, 1, 17, 1024] {
            for workers in [0, 1, 4, 16] {
                let a = WeightAllocation::split(total, workers, "test");
                assert!(a.main_bytes + a.worker_bytes * workers <= total);
                if workers == 0 {
                    assert_eq!(a.main_bytes, total);
                }
            }
        }
    }
}
