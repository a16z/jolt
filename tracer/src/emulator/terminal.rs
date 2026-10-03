pub trait Terminal {
    fn put_byte(&mut self, value: u8);

    /// Gets an output ascii byte data from output buffer.
    /// This method returns zero if the buffer is empty.
    fn get_output(&mut self) -> u8;

    fn put_input(&mut self, data: u8);

    fn get_input(&mut self) -> u8;
}

#[derive(Default)]
pub struct DummyTerminal {}

impl Terminal for DummyTerminal {
    fn put_byte(&mut self, _value: u8) {}
    fn get_output(&mut self) -> u8 {
        0
    }
    fn put_input(&mut self, _value: u8) {}
    fn get_input(&mut self) -> u8 {
        0
    }
}
