use std::thread::{self, JoinHandle};

pub fn drop_in_background_thread<T>(data: T) where T: Send + 'static {
    // TODO(sragss): Add tracing to these drops
    rayon::spawn(move || drop(data));
    // std::mem::forget(data);
}

pub fn allocate_vec_in_background<T: Clone + Send + 'static>(value: T, size: usize) -> JoinHandle<Vec<T>> {
    thread::spawn(move || vec![value; size])
}

pub fn allocate_zero_vec_background<T: Clone + Send + 'static>(size: usize) -> JoinHandle<Vec<T>> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value
    thread::spawn(move || {
        let result: Vec<T>;

        unsafe {
            let layout = std::alloc::Layout::array::<T>(size).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
    
            if ptr.is_null() {
                panic!("Allocation failed");
            }
    
            result = Vec::from_raw_parts(ptr, size, size);
        }
        result
    })
}
