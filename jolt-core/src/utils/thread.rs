use std::thread::{self, JoinHandle};

pub fn drop_in_background_thread<T>(data: T) where T: Send + 'static {
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

pub fn allocate_vec_in_background<T: Clone + Send + 'static>(value: T, size: usize) -> JoinHandle<Vec<T>> {
    thread::spawn(move || vec![value; size])
}