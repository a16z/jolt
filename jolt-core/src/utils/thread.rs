use std::thread;

pub fn drop_in_background_thread<T>(data: T) where T: Send + 'static {
    thread::spawn(move || drop(data));
}
