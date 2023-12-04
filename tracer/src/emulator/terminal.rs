/// Emulates terminal. It holds input/output data in buffer
/// transferred to/from `Emulator`.
pub trait Terminal {
    /// Puts an output ascii byte data to output buffer.
    /// The data is expected to be read by user program via `get_output()`
    /// and be displayed to user.
    fn put_byte(&mut self, value: u8);

    /// Gets an output ascii byte data from output buffer.
    /// This method returns zero if the buffer is empty.
    fn get_output(&mut self) -> u8;

    /// Puts an input ascii byte data to input buffer.
    /// The data is expected to be read by `Emulator` via `get_input()`
    /// and be handled.
    fn put_input(&mut self, data: u8);

    /// Gets an input ascii byte data from input buffer.
    /// Used by `Emulator`.
    fn get_input(&mut self) -> u8;
}

/// For the test or whatever.
pub struct DummyTerminal {}

impl DummyTerminal {
    pub fn new() -> Self {
        DummyTerminal {}
    }
}

impl Terminal for DummyTerminal {
    fn put_byte(&mut self, _value: u8) {}
    fn get_input(&mut self) -> u8 {
        0
    }
    fn put_input(&mut self, _value: u8) {}
    fn get_output(&mut self) -> u8 {
        0
    }
}
