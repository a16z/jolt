#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable]
fn mandelbrot(cx: f64, cy: f64, max_iter: usize) -> bool {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut iter = 0;

    while x*x + y*y <= 4.0 && iter < max_iter {
        let temp_x = x*x - y*y + cx;
        y = 2.0*x*y + cy;
        x = temp_x;
        iter += 1;
    }

    // If we've reached max_iter without diverging, assume in the set
    iter == max_iter
}