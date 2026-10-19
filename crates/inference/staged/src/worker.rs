//! Private line protocol for serial jobs in long-lived auxiliary processes.
use crate::cache::CachedInputs;
use crate::chain_runner::StageEnvGuard;
use crate::routines::{self, DirectTimings, StageKind};
use crate::weights::RunWeights;
use anyhow::{bail, Context, Result};
use inference_kernels::weight_budget::WeightCacheStats;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

const VERSION: u32 = 1;
const MAX_LINE: u64 = 1 << 20;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerJob {
    pub id: usize,
    pub name: String,
    pub kind: StageKind,
    pub input: PathBuf,
    pub input_manifest: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Serialize, Deserialize)]
enum Request {
    Run { version: u32, job: WorkerJob },
    Shutdown { version: u32 },
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerResponse {
    version: u32,
    pub id: usize,
    pub name: String,
    pub timings: Option<DirectTimings>,
    pub cache: WeightCacheStats,
    pub error: Option<String>,
}

fn read_message<T: serde::de::DeserializeOwned>(reader: &mut impl BufRead) -> Result<Option<T>> {
    let mut line = String::new();
    let size = reader.take(MAX_LINE + 1).read_line(&mut line)?;
    if size == 0 {
        return Ok(None);
    }
    if size as u64 > MAX_LINE || !line.ends_with('\n') {
        bail!("invalid worker protocol frame length");
    }
    Ok(Some(
        serde_json::from_str(&line).context("invalid worker protocol JSON")?,
    ))
}

fn write_message(writer: &mut impl Write, message: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, message)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Called only by the hidden entrypoint. stdout is reserved for protocol frames.
pub fn run_worker(budget_bytes: usize) -> Result<()> {
    crate::ple::Loading::from_env()?;
    let weights = RunWeights::new(budget_bytes);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    while let Some(request) = read_message::<Request>(&mut reader)? {
        match request {
            Request::Shutdown { version } => {
                if version != VERSION {
                    bail!("unsupported worker protocol version {version}");
                }
                return Ok(());
            }
            Request::Run { version, job } => {
                if version != VERSION {
                    bail!("unsupported worker protocol version {version}");
                }
                let result = execute_job(&job, &weights);
                let failed = result.is_err();
                let (timings, error) = match result {
                    Ok(timings) => (Some(timings), None),
                    Err(error) => (None, Some(format!("{}: {error:#}", job.name))),
                };
                write_message(
                    &mut writer,
                    &WorkerResponse {
                        version: VERSION,
                        id: job.id,
                        name: job.name,
                        timings,
                        cache: weights.stats(),
                        error,
                    },
                )?;
                // A failed stage may have partially initialized runtime state. Never reuse it.
                if failed {
                    bail!("auxiliary worker stage failed");
                }
            }
        }
    }
    Ok(())
}

fn execute_job(job: &WorkerJob, weights: &RunWeights) -> Result<DirectTimings> {
    if !matches!(job.kind, StageKind::PrefillPrepareAux { .. }) {
        bail!("worker accepts only auxiliary stages");
    }
    if StageKind::from_stage_name(&job.name)? != job.kind {
        bail!("worker stage name/kind mismatch");
    }
    let _env = StageEnvGuard::apply(&job.output_dir);
    let result = routines::run_and_publish_from_paths_with_caches(
        &job.kind,
        &job.input,
        &job.input_manifest,
        &CachedInputs::new(),
        weights.caches(),
    )?;
    Ok(result.timings)
}

struct Worker {
    child: Arc<Mutex<Child>>,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Worker {
    fn spawn(exe: &Path, budget: usize) -> Result<Self> {
        let mut command = Command::new(exe);
        command
            .arg("--aux-worker")
            .arg(budget.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command
            .spawn()
            .context("failed to start auxiliary worker")?;
        let input = child.stdin.take().expect("piped stdin");
        let output = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            input,
            output,
        })
    }
    fn run(&mut self, job: WorkerJob) -> Result<WorkerResponse> {
        let id = job.id;
        let name = job.name.clone();
        write_message(
            &mut self.input,
            &Request::Run {
                version: VERSION,
                job,
            },
        )
        .with_context(|| format!("failed to submit auxiliary stage {name}"))?;
        let response = read_message::<WorkerResponse>(&mut self.output)
            .with_context(|| format!("invalid auxiliary response for {name}"))?
            .with_context(|| format!("auxiliary worker exited before completing {name}"))?;
        if response.version != VERSION || response.id != id || response.name != name {
            bail!("auxiliary response identity/version mismatch for {name}");
        }
        if let Some(error) = &response.error {
            bail!("auxiliary stage {name} failed: {error}");
        }
        if response.timings.is_none() {
            bail!("auxiliary stage {name} omitted timings");
        }
        Ok(response)
    }
}

pub(crate) struct WorkerPool {
    workers: Vec<Worker>,
}

impl WorkerPool {
    pub fn new(exe: &Path, count: usize, budget: usize) -> Result<Self> {
        let mut pool = Self {
            workers: Vec::new(),
        };
        for _ in 0..count {
            pool.workers.push(Worker::spawn(exe, budget)?);
        }
        Ok(pool)
    }
    pub fn len(&self) -> usize {
        self.workers.len()
    }
    pub fn process_ids(&self) -> Vec<u32> {
        self.workers
            .iter()
            .map(|w| w.child.lock().unwrap().id())
            .collect()
    }
    pub fn run(&mut self, jobs: Vec<WorkerJob>) -> Result<Vec<WorkerResponse>> {
        let mut groups: Vec<Vec<WorkerJob>> = (0..self.len()).map(|_| Vec::new()).collect();
        for job in jobs {
            let StageKind::PrefillPrepareAux { layer } = job.kind else {
                bail!("non-auxiliary pool job");
            };
            groups[layer % self.len()].push(job);
        }
        let children: Vec<_> = self.workers.iter().map(|w| Arc::clone(&w.child)).collect();
        std::thread::scope(|scope| {
            let (sender, receiver) = std::sync::mpsc::channel();
            for (worker, jobs) in self.workers.iter_mut().zip(groups) {
                let sender = sender.clone();
                scope.spawn(move || {
                    // Report panics as failures too, so peers are killed before joining.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        jobs.into_iter()
                            .map(|job| worker.run(job))
                            .collect::<Result<Vec<_>>>()
                    }))
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("auxiliary protocol thread panicked")));
                    let _ = sender.send(result);
                });
            }
            drop(sender);
            let mut results = Vec::new();
            let mut failure = None;
            for result in receiver {
                match result {
                    Ok(mut batch) => results.append(&mut batch),
                    Err(error) => {
                        if failure.is_none() {
                            failure = Some(error);
                            for child in &children {
                                let _ = child.lock().unwrap().kill();
                            }
                        }
                    }
                }
            }
            if let Some(error) = failure {
                return Err(error);
            }
            results.sort_by_key(|r| r.id);
            Ok(results)
        })
    }
    pub fn shutdown(&mut self) -> Result<()> {
        for worker in &mut self.workers {
            write_message(&mut worker.input, &Request::Shutdown { version: VERSION })?;
        }
        for worker in &mut self.workers {
            let started = std::time::Instant::now();
            loop {
                if let Some(status) = worker.child.lock().unwrap().try_wait()? {
                    if !status.success() {
                        bail!("auxiliary worker shutdown failed: {status}");
                    }
                    break;
                }
                if started.elapsed() > std::time::Duration::from_secs(2) {
                    bail!("auxiliary worker did not shut down");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        Ok(())
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let mut child = worker.child.lock().unwrap();
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_rejects_truncation_and_invalid_json() {
        for bytes in [b"{}".as_slice(), b"not-json\n", b"{}\n"] {
            assert!(read_message::<Request>(&mut std::io::Cursor::new(bytes)).is_err());
        }
        assert!(
            read_message::<Request>(&mut std::io::Cursor::new(Vec::<u8>::new()))
                .unwrap()
                .is_none()
        );
        assert!(read_message::<Request>(&mut std::io::Cursor::new(vec![
            b'x';
            MAX_LINE as usize + 1
        ]))
        .is_err());
    }

    #[test]
    #[cfg(unix)]
    fn pool_reaps_dead_workers_on_error() {
        let mut pool = WorkerPool::new(Path::new("/bin/false"), 2, 0).unwrap();
        let children: Vec<_> = pool.workers.iter().map(|w| Arc::clone(&w.child)).collect();
        let result = pool.run(vec![WorkerJob {
            id: 0,
            name: "prefill_prepare_aux_l0".into(),
            kind: StageKind::PrefillPrepareAux { layer: 0 },
            input: "missing".into(),
            input_manifest: "missing".into(),
            output_dir: "missing".into(),
        }]);
        assert!(result.is_err());
        drop(pool);
        for child in children {
            assert!(child.lock().unwrap().try_wait().unwrap().is_some());
        }
    }
}
