use crate::expand::allocator::ExpansionAllocator;

#[derive(Debug)]
pub struct InstrAssembler<'a> {
    pub allocator: &'a mut ExpansionAllocator,
}
