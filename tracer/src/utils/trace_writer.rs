use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    path::Path,
    sync::mpsc::{sync_channel, Receiver, SyncSender},
    thread::{self, JoinHandle},
    time::Instant,
};

// batches of size 5M (~400MB) RV32IMCycles (80 bytes each)
const BATCH_SIZE: usize = 5_000_000;
// total memory usage of the channel is BATCH * CHANNEL_DEPTH = ~2 GB
const CHANNEL_DEPTH: usize = 64;

/// Configuration for the trace writer
#[derive(Debug, Clone)]
pub struct TraceWriterConfig {
    /// Size of each batch in number of items
    pub batch_size: usize,
    /// Depth of the channel (number of batches that can be queued)
    pub channel_depth: usize,
    /// Buffer size for the file writer (in bytes)
    pub write_buffer_size: usize,
    /// Threshold in milliseconds for logging slow batch sends
    pub slow_batch_threshold_ms: u128,
}

impl Default for TraceWriterConfig {
    /// Defaults were chosen to prefer performance on a M4 Max for traces containing RV32IMCycle
    fn default() -> Self {
        Self {
            batch_size: BATCH_SIZE,
            channel_depth: CHANNEL_DEPTH,
            write_buffer_size: 1 << 26,
            slow_batch_threshold_ms: 150,
        }
    }
}

/// A generic trace writer that handles batched writing to files
pub struct TraceWriter<T> {
    sender: Option<SyncSender<Vec<T>>>,
    writer_handle: Option<JoinHandle<std::io::Result<()>>>,
    config: TraceWriterConfig,
    _phantom: PhantomData<T>,
}

impl<T> TraceWriter<T>
where
    T: serde::Serialize + Send + 'static,
{
    /// Create a new TraceWriter with the given configuration
    pub fn new(output_path: impl AsRef<Path>, config: TraceWriterConfig) -> std::io::Result<Self> {
        let (sender, receiver) = sync_channel::<Vec<T>>(config.channel_depth);

        let writer_handle = Self::spawn_writer_thread(
            output_path.as_ref().to_path_buf(),
            receiver,
            config.write_buffer_size,
        );

        Ok(Self {
            sender: Some(sender),
            writer_handle: Some(writer_handle),
            config,
            _phantom: PhantomData,
        })
    }

    /// Create a new TraceWriter with default configuration
    pub fn with_defaults(output_path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::new(output_path, TraceWriterConfig::default())
    }

    /// Spawn the background writer thread
    fn spawn_writer_thread(
        path: std::path::PathBuf,
        receiver: Receiver<Vec<T>>,
        buffer_size: usize,
    ) -> JoinHandle<std::io::Result<()>> {
        thread::spawn(move || -> std::io::Result<()> {
            let file = File::create(&path)?;
            let mut buf = BufWriter::with_capacity(buffer_size, file);

            while let Ok(chunk) = receiver.recv() {
                postcard::to_io(&chunk, &mut buf).map_err(std::io::Error::other)?;
            }

            buf.flush()?;
            Ok(())
        })
    }

    /// Send a batch to be written
    /// Returns true if the batch was sent successfully
    pub fn send_batch(&self, batch: Vec<T>) -> bool {
        if let Some(sender) = &self.sender {
            let start = Instant::now();

            if sender.send(batch).is_ok() {
                let elapsed = start.elapsed().as_millis();
                if elapsed > self.config.slow_batch_threshold_ms {
                    eprintln!(
                        "Slow batch send: {} items in {:.2}ms",
                        self.config.batch_size, elapsed
                    );
                }
                return true;
            }
        }
        false
    }

    /// Finalize the writer and wait for all pending writes to complete
    pub fn finalize(mut self) -> std::io::Result<()> {
        // Drop the sender to signal the writer thread to finish
        self.sender.take();

        let start = Instant::now();

        if let Some(handle) = self.writer_handle.take() {
            let result = handle
                .join()
                .map_err(|_| std::io::Error::other("Writer thread panicked"))?;

            println!(
                "Writer thread finished in {:.2}ms",
                start.elapsed().as_millis()
            );

            result
        } else {
            Ok(())
        }
    }
}

pub struct TraceBatchCollector<T> {
    writer: TraceWriter<T>,
    current_batch: Vec<T>,
    total_items: usize,
}

impl<T> TraceBatchCollector<T>
where
    T: serde::Serialize + Send + 'static,
{
    pub fn new(writer: TraceWriter<T>) -> Self {
        let batch_capacity = writer.config.batch_size;
        Self {
            writer,
            current_batch: Vec::with_capacity(batch_capacity),
            total_items: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        self.current_batch.push(item);
        self.total_items += 1;

        if self.current_batch.len() >= self.writer.config.batch_size {
            self.flush_batch();
        }
    }

    fn flush_batch(&mut self) {
        if !self.current_batch.is_empty() {
            let batch = std::mem::replace(
                &mut self.current_batch,
                Vec::with_capacity(self.writer.config.batch_size),
            );

            if !self.writer.send_batch(batch) {
                eprintln!("Failed to send batch to writer");
            }
        }
    }

    /// Finalize the collector, flushing any remaining items
    pub fn finalize(mut self) -> std::io::Result<usize> {
        self.flush_batch();
        self.writer.finalize()?;
        Ok(self.total_items)
    }

    pub fn total_items(&self) -> usize {
        self.total_items
    }
}
