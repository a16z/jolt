use std::num::NonZeroUsize;

/// A streaming visitor: false stops before reading another shape member.
pub trait KeySink {
    fn word(&mut self, value: u64) -> bool;
}

pub trait KeyVisit {
    fn visit(&self, sink: &mut dyn KeySink) -> bool;
}

impl KeyVisit for usize {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        u64::try_from(*self)
            .ok()
            .is_some_and(|word| sink.word(word))
    }
}

impl KeyVisit for bool {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        sink.word(u64::from(*self))
    }
}

impl KeyVisit for NonZeroUsize {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        self.get().visit(sink)
    }
}

impl<T: KeyVisit> KeyVisit for Vec<T> {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        self.len().visit(sink) && self.iter().all(|value| value.visit(sink))
    }
}

impl KeyVisit for () {
    fn visit(&self, _sink: &mut dyn KeySink) -> bool {
        true
    }
}

#[macro_export]
macro_rules! recipe_shape {
    ($ty:ty { $($field:ident),* $(,)? }) => {
        impl $crate::immutable_recipes::KeyVisit for $ty {
            fn visit(&self, sink: &mut dyn $crate::immutable_recipes::KeySink) -> bool {
                let Self { $($field),* } = self;
                true $(&& $crate::immutable_recipes::KeyVisit::visit($field, sink))*
            }
        }
    };
    ($ty:ty => ( $field:ident )) => {
        impl $crate::immutable_recipes::KeyVisit for $ty {
            fn visit(&self, sink: &mut dyn $crate::immutable_recipes::KeySink) -> bool {
                let Self($field) = self;
                $crate::immutable_recipes::KeyVisit::visit($field, sink)
            }
        }
    };
}

#[macro_export]
macro_rules! recipe_id {
    ($ty:ty { $( $(#[$attr:meta])* $variant:ident
        $( ( $($value:ident : $tuple:ty),* $(,)? ) )?
        $( { $($field:ident : $named:ty),* $(,)? } )? = $tag:literal ),* $(,)? }) => {
        impl $crate::immutable_recipes::KeyVisit for $ty {
            fn visit(&self, sink: &mut dyn $crate::immutable_recipes::KeySink) -> bool {
                match self {
                    $( $(#[$attr])* Self::$variant $( ($($value),*) )? $( { $($field),* } )?
                        => sink.word($tag)
                            $( $(&& $crate::immutable_recipes::KeyVisit::visit($value, sink))* )?
                            $( $(&& $crate::immutable_recipes::KeyVisit::visit($field, sink))* )?, )*
                }
            }
        }

    };
}
